"""
HDF5 compatibility layer for PyTorch featurization.

This module provides functions to write PyTorch-generated features to HDF5
files in the format expected by TrnSet, making it a drop-in replacement for
the Fortran-based featurization workflow.
"""

import glob
import os
from typing import Dict, List, Optional, Union

import numpy as np
import tables as tb
import torch

from ..formats.xsf import XSFParser
from ..geometry import AtomicStructure
from ..trainset import FeaturizedAtomicStructure
from .featurize import ChebyshevDescriptor

__author__ = "The aenet developers"
__date__ = "2025-01-22"


class TorchAUCFeaturizer:
    """
    PyTorch-based AUC featurizer with HDF5 output compatibility.

    This class provides the same API as AenetAUCFeaturizer but uses the
    pure Python/PyTorch implementation instead of Fortran executables.

    Attributes
    ----------
        typenames: List of atomic species
        rad_order: Radial Chebyshev polynomial order
        rad_cutoff: Radial cutoff radius (Angstroms)
        ang_order: Angular Chebyshev polynomial order
        ang_cutoff: Angular cutoff radius (Angstroms)
        min_cutoff: Minimum distance cutoff (Angstroms)
        device: 'cpu' or 'cuda'
        dtype: torch.float64 for double precision
    """

    def __init__(
        self,
        typenames: List[str],
        rad_order: int = 0,
        rad_cutoff: float = 0.0,
        ang_order: int = 0,
        ang_cutoff: float = 0.0,
        min_cutoff: float = 0.55,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        **kwargs
    ):
        """
        Initialize PyTorch-based AUC featurizer.

        Args:
            typenames: List of atomic species (e.g., ['O', 'H'])
            rad_order: Maximum radial Chebyshev order
            rad_cutoff: Radial cutoff radius (Angstroms)
            ang_order: Maximum angular Chebyshev order
            ang_cutoff: Angular cutoff radius (Angstroms)
            min_cutoff: Minimum distance cutoff (Angstroms)
            device: 'cpu' or 'cuda'
            dtype: torch.float64 for double precision
        """
        self.typenames = typenames
        self.rad_order = rad_order
        self.rad_cutoff = rad_cutoff
        self.ang_order = ang_order
        self.ang_cutoff = ang_cutoff
        self.min_cutoff = min_cutoff
        self.device = device
        self.dtype = dtype

        # Create descriptor
        self.descriptor = ChebyshevDescriptor(
            species=typenames,
            rad_order=rad_order,
            rad_cutoff=rad_cutoff,
            ang_order=ang_order,
            ang_cutoff=ang_cutoff,
            min_cutoff=min_cutoff,
            device=device,
            dtype=dtype,
        )

        for arg in kwargs:
            print(f"Warning: unknown keyword `{arg}`")

    @property
    def ntypes(self):
        """Number of atomic species."""
        return len(self.typenames)

    def _featurize_structure_dict(
        self,
        struc: AtomicStructure,
        **kwargs
    ) -> Dict:
        """
        Internal method to featurize structure and return dict.

        Args:
            struc: AtomicStructure instance

        Returns
        -------
            Dictionary with featurization results
        """
        # Get structure data
        positions = struc.coords[-1]
        species = struc.types

        # Handle energy - check if it's set and not None
        energy = struc.energy[-1] if (struc.energy[-1] is not None) else 0.0

        # Handle forces - check if set, not None, and not empty array
        forces_data = struc.forces[-1] if len(struc.forces) > 0 else None
        if forces_data is not None and len(forces_data) == 0:
            forces_data = None
        forces = forces_data

        # Handle PBC
        cell = struc.avec[-1] if struc.pbc else None
        pbc = np.array([True, True, True]) if struc.pbc else None

        # Convert to torch tensors
        positions_torch = torch.from_numpy(positions).to(self.dtype)
        cell_torch = torch.from_numpy(
            cell).to(self.dtype) if cell is not None else None
        pbc_torch = torch.from_numpy(pbc) if pbc is not None else None

        # Featurize
        with torch.no_grad():
            features = self.descriptor(
                positions_torch, species, cell_torch, pbc_torch
            ).cpu().numpy()

        # Prepare atomic data
        atoms = []
        for i, (sp, coords, feat) in enumerate(zip(
                                          species, positions, features)):
            atom_data = {
                "type": sp,
                "fingerprint": feat,
                "coords": coords,
                "forces": forces[i] if forces is not None else np.zeros(3),
            }
            atoms.append(atom_data)

        return {
            "path": getattr(struc, 'path', 'unknown'),
            "energy": energy,
            "atom_types": self.typenames,
            "atoms": atoms,
        }

    def featurize_structure(
        self,
        struc: AtomicStructure,
        **kwargs
    ) -> FeaturizedAtomicStructure:
        """
        Featurize a single atomic structure.

        Args:
            struc: AtomicStructure instance

        Returns
        -------
            FeaturizedAtomicStructure object with .atom_features property
        """
        # Get dict representation
        feat_dict = self._featurize_structure_dict(struc, **kwargs)

        # Convert to FeaturizedAtomicStructure
        return FeaturizedAtomicStructure(
            path=feat_dict["path"],
            energy=feat_dict["energy"],
            atom_types=self.typenames,
            atoms=feat_dict["atoms"]
        )

    def featurize_structures(
        self,
        structures: List[AtomicStructure],
        **kwargs
    ) -> List[FeaturizedAtomicStructure]:
        """
        Featurize multiple atomic structures.

        Args:
            structures: List of AtomicStructure instances

        Returns
        -------
            List of FeaturizedAtomicStructure objects
        """
        return [self.featurize_structure(s, **kwargs) for s in structures]

    def run_aenet_generate(
        self,
        xsf_files: Union[List[os.PathLike], str],
        workdir: Optional[os.PathLike] = None,
        hdf5_filename: str = 'features.h5',
        output_file: str = 'generate.out',
        atomic_energies: Optional[Dict[str, float]] = None,
        debug: bool = False,
        **kwargs
    ):
        """
        Featurize structures and write to HDF5 (drop-in replacement
        for the Fortran version).

        This method provides the same API as
        AenetAUCFeaturizer.run_aenet_generate() but uses the PyTorch
        implementation instead of Fortran executables.

        Args:
            xsf_files: List of XSF file paths or glob pattern
            workdir: Working directory (not used in PyTorch version,
              kept for API compatibility)
            hdf5_filename: Name of output HDF5 file
            output_file: Name of output log file (for API compatibility)
            atomic_energies: Dictionary of atomic energies {species: energy}
            debug: Enable debug output
            **kwargs: Additional arguments (forwarded for compatibility)
        """
        # Handle glob pattern
        if isinstance(xsf_files, str):
            xsf_files = glob.glob(xsf_files)

        # Set default atomic energies
        if atomic_energies is None:
            atomic_energies = {sp: 0.0 for sp in self.typenames}

        # Read structures
        print(f"Reading {len(xsf_files)} structure files...")
        xsf_parser = XSFParser()
        structures = []

        for i, xsf_file in enumerate(xsf_files):
            if not os.path.exists(xsf_file):
                raise FileNotFoundError(f"File not found: {xsf_file}")

            struc = xsf_parser.read(xsf_file)
            # Store the file path for reference
            struc.path = os.path.abspath(xsf_file)

            # Ensure energy and forces are set
            if struc.energy[-1] is None:
                struc.energy[-1] = 0.0
            if struc.forces[-1] is None or len(struc.forces[-1]) == 0:
                struc.forces[-1] = np.zeros(struc.coords[-1].shape)

            structures.append(struc)

            if (i + 1) % 100 == 0:
                print(f"  Read {i + 1}/{len(xsf_files)} structures")

        print(f"Featurizing {len(structures)} structures...")

        # Featurize all structures (use internal dict method for HDF5)
        featurized_structures = []
        for i, struc in enumerate(structures):
            feat_struc = self._featurize_structure_dict(struc)
            featurized_structures.append(feat_struc)

            if (i + 1) % 100 == 0:
                print(f"  Featurized {i + 1}/{len(structures)} structures")

        # Write to HDF5
        print(f"Writing features to {hdf5_filename}...")
        write_features_to_hdf5(
            featurized_structures=featurized_structures,
            filename=hdf5_filename,
            typenames=self.typenames,
            atomic_energies=atomic_energies,
            name="PyTorch-generated training set",
        )

        # Write output file for compatibility
        if output_file and workdir:
            os.makedirs(workdir, exist_ok=True)
            output_path = os.path.join(workdir, output_file)
        elif output_file:
            output_path = output_file
        else:
            output_path = None

        if output_path:
            with open(output_path, 'w') as f:
                f.write("PyTorch-based featurization completed successfully\n")
                f.write(f"Processed {len(structures)} structures\n")
                f.write(f"Output file: {hdf5_filename}\n")

        print(f"Done! Features written to {hdf5_filename}")


def write_features_to_hdf5(
    featurized_structures: List[Dict],
    filename: os.PathLike,
    typenames: List[str],
    atomic_energies: Dict[str, float],
    name: str = "Training set",
    normalized: bool = False,
    scale: float = 1.0,
    shift: float = 0.0,
    complevel: int = 1,
):
    """
    Write featurized structures to HDF5 file in TrnSet format.

    Args:
        featurized_structures: List of featurization result dictionaries
        filename: Output HDF5 file path
        typenames: List of atomic species
        atomic_energies: Dictionary of atomic energies
        name: Dataset name
        normalized: Whether features are normalized
        scale: Feature scaling factor
        shift: Feature shift value
        complevel: HDF5 compression level (0-9)
    """
    # Compute statistics
    num_atoms_tot = sum(len(s["atoms"]) for s in featurized_structures)
    num_structures = len(featurized_structures)

    # Normalize energies by subtracting atomic reference energies
    # and dividing by number of atoms (matching Fortran behavior)
    normalized_energies = []
    for s in featurized_structures:
        total_energy = s["energy"]
        num_atoms = len(s["atoms"])
        # Subtract atomic energy contributions
        atomic_contribution = sum(
            atomic_energies.get(atom["type"], 0.0)
            for atom in s["atoms"]
        )
        # Normalize per atom
        normalized_energy = (total_energy - atomic_contribution) / num_atoms
        normalized_energies.append(normalized_energy)

    E_min = min(normalized_energies)
    E_max = max(normalized_energies)
    E_av = np.mean(normalized_energies)

    # Prepare atomic energy array
    atomic_energy_array = np.array(
        [atomic_energies.get(t, 0.0) for t in typenames]
    )

    # Create HDF5 file
    h5file = tb.open_file(filename, mode='w', title='Aenet reference data')

    try:
        # Create groups
        structures = h5file.create_group(
            h5file.root, "structures", "Atomic structures"
        )

        # Write metadata
        n_types = len(typenames)
        metadata = h5file.create_table(
            h5file.root, "metadata", {
                'name': tb.StringCol(itemsize=1024),
                'normalized': tb.BoolCol(),
                'scale': tb.Float64Col(),
                'shift': tb.Float64Col(),
                'atom_types': tb.StringCol(itemsize=64, shape=(n_types,)),
                'atomic_energy': tb.Float64Col(shape=(n_types,)),
                'num_atoms_tot': tb.UInt64Col(),
                'num_structures': tb.UInt64Col(),
                'E_min': tb.Float64Col(),
                'E_max': tb.Float64Col(),
                'E_av': tb.Float64Col()
            },
            "General information about the data set"
        )

        metadata.row['name'] = name
        metadata.row['normalized'] = normalized
        metadata.row['scale'] = scale
        metadata.row['shift'] = shift
        metadata.row['atom_types'] = typenames
        metadata.row['atomic_energy'] = atomic_energy_array
        metadata.row['num_atoms_tot'] = num_atoms_tot
        metadata.row['num_structures'] = num_structures
        metadata.row['E_min'] = E_min
        metadata.row['E_max'] = E_max
        metadata.row['E_av'] = E_av
        metadata.row.append()

        # Create structure tables
        info_table_dict = {
            "path": tb.StringCol(itemsize=1024),
            "first_atom": tb.UInt64Col(),
            "num_atoms": tb.UInt32Col(),
            "energy": tb.Float64Col()
        }
        atom_table_dict = {
            "structure": tb.UInt64Col(),
            "type": tb.StringCol(itemsize=64),
            "coords": tb.Float64Col(shape=(3,)),
            "forces": tb.Float64Col(shape=(3,))
        }

        info = h5file.create_table(
            structures, "info", info_table_dict,
            "Atomic structure information",
            tb.Filters(complevel, shuffle=False)
        )
        atoms = h5file.create_table(
            structures, "atoms", atom_table_dict,
            "Atomic data",
            tb.Filters(complevel, shuffle=False)
        )
        features = h5file.create_vlarray(
            structures, "features", tb.Float64Atom(),
            "Atomic environment features",
            tb.Filters(complevel, shuffle=False)
        )

        # Write structure data
        iatom = 0
        for i, s in enumerate(featurized_structures):
            # Compute cohesive energy (total - atomic contributions)
            # matching Fortran behavior
            total_energy = s['energy']
            atomic_contribution = sum(
                atomic_energies.get(atom["type"], 0.0)
                for atom in s["atoms"]
            )
            cohesive_energy = total_energy - atomic_contribution

            info.row['path'] = s['path']
            info.row['first_atom'] = iatom
            info.row['num_atoms'] = len(s['atoms'])
            info.row['energy'] = cohesive_energy
            info.row.append()

            for j, atom in enumerate(s['atoms']):
                atoms.row['structure'] = i
                atoms.row['type'] = atom['type']
                atoms.row['coords'] = atom['coords']
                atoms.row['forces'] = atom['forces']
                atoms.row.append()
                features.append(atom['fingerprint'])

            iatom += len(s['atoms'])

        h5file.flush()

    finally:
        h5file.close()


def featurize_and_write_hdf5(
    xsf_files: Union[List[os.PathLike], str],
    typenames: List[str],
    rad_order: int,
    rad_cutoff: float,
    ang_order: int,
    ang_cutoff: float,
    hdf5_filename: str = 'features.h5',
    atomic_energies: Optional[Dict[str, float]] = None,
    min_cutoff: float = 0.55,
    device: str = "cpu",
    **kwargs
):
    """
    Convenience function to featurize structures and write to HDF5.

    Args:
        xsf_files: List of XSF file paths or glob pattern
        typenames: List of atomic species
        rad_order: Radial Chebyshev polynomial order
        rad_cutoff: Radial cutoff radius (Angstroms)
        ang_order: Angular Chebyshev polynomial order
        ang_cutoff: Angular cutoff radius (Angstroms)
        hdf5_filename: Output HDF5 file path
        atomic_energies: Dictionary of atomic energies
        min_cutoff: Minimum distance cutoff
        device: 'cpu' or 'cuda'
    """
    featurizer = TorchAUCFeaturizer(
        typenames=typenames,
        rad_order=rad_order,
        rad_cutoff=rad_cutoff,
        ang_order=ang_order,
        ang_cutoff=ang_cutoff,
        min_cutoff=min_cutoff,
        device=device,
        **kwargs
    )

    featurizer.run_aenet_generate(
        xsf_files=xsf_files,
        hdf5_filename=hdf5_filename,
        atomic_energies=atomic_energies,
    )
