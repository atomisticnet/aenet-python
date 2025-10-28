import math
from pathlib import Path

import numpy as np
import torch

from aenet.torch_training import TorchANNPotential, TorchTrainingConfig, Structure
from aenet.torch_featurize import ChebyshevDescriptor
from aenet.formats.xsf import XSFParser


def load_tio2_structures(n_structs: int = 10):
    xsf_dir = Path("notebooks/xsf-TiO2")
    files = sorted(xsf_dir.glob("*.xsf"))[:n_structs]
    if not files:
        raise RuntimeError(f"No XSF files found under {xsf_dir.resolve()}")
    parser = XSFParser()
    out = []
    for p in files:
        s = parser.read(str(p))
        positions = s.coords[-1]
        species = s.types
        energy = float(s.energy[-1]) if (s.energy and s.energy[-1] is not None) else 0.0
        cell = np.array(s.avec[-1]) if s.pbc else None
        pbc = np.array([True, True, True]) if s.pbc else None
        out.append(
            Structure(
                positions=positions,
                species=species,
                energy=energy,
                forces=None,
                cell=cell,
                pbc=pbc,
                name=p.name,
            )
        )
    return out


def main():
    print(f"torch version: {torch.__version__}")

    structures = load_tio2_structures(n_structs=10)
    print(f"Loaded {len(structures)} structures; first has {structures[0].n_atoms} atoms")

    descriptor = ChebyshevDescriptor(
        species=["Ti", "O"],
        rad_order=3,
        rad_cutoff=5.0,
        ang_order=1,
        ang_cutoff=3.0,
        min_cutoff=0.5,
        device="cpu",
        dtype=torch.float64,
    )

    arch = {
        "Ti": [(10, "tanh"), (10, "tanh")],
        "O": [(10, "tanh"), (10, "tanh")],
    }

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=20,
        testpercent=10,
        force_weight=0.0,  # energy-only
        memory_mode="cpu",
        device="cpu",
        energy_target="cohesive",
        E_atomic={"Ti": -1604.604515075, "O": -432.503149303},
        # normalization defaults are enabled (features + energy)
    )

    history = pot.train(
        structures=structures,
        config=cfg,
        checkpoint_dir="notebooks/example-05-outputs/ckpts",
        checkpoint_interval=0,
        max_checkpoints=0,
        save_best=False,
        use_scheduler=False,
    )

    tr_hist = history.get("train_energy_rmse", [])
    te_hist = history.get("test_energy_rmse", [])
    tr_last = tr_hist[-1] if tr_hist else float("nan")
    te_last = te_hist[-1] if te_hist else float("nan")
    print(f"Final train_energy_rmse: {tr_last:.6f} eV/atom")
    if not math.isnan(te_last):
        print(f"Final test_energy_rmse:  {te_last:.6f} eV/atom")

    # Sanity print of normalization internals
    print("Normalization:")
    print(f"  feature_norm: {getattr(pot, '_normalize_features', False)}")
    print(f"  energy_norm:  {getattr(pot, '_normalize_energy', False)}")
    print(f"  E_shift(per-atom): {getattr(pot, '_E_shift', None)}")
    print(f"  E_scaling:         {getattr(pot, '_E_scaling', None)}")


if __name__ == "__main__":
    main()
