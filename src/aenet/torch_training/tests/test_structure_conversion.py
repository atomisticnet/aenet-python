import numpy as np
import pytest

from aenet.geometry import AtomicStructure
from aenet.torch_training.config import Structure as TorchStructure


def assert_array_equal(a, b, tol=0.0):
    a = np.array(a)
    b = np.array(b)
    if tol == 0.0:
        assert np.array_equal(a, b), f"Arrays differ:\n{a}\n!=\n{b}"
    else:
        assert np.allclose(a, b, atol=tol), f"Arrays differ:\n{a}\n!=\n{b}"


def test_periodic_with_forces_roundtrip():
    # Periodic, with energy and forces
    coords = np.array([[0.1, 0.2, 0.3],
                       [1.0, 1.1, 1.2]])
    types = ["H", "O"]
    cell = np.array([[10.0, 0.0, 0.0],
                     [0.0, 10.0, 0.0],
                     [0.0, 0.0, 10.0]])
    energy = -1.2345
    forces = np.array([[0.01, 0.02, 0.03],
                       [-0.01, -0.02, -0.03]])

    a = AtomicStructure(coords=coords, types=types, avec=cell,
                        fractional=False, energy=energy, forces=forces)

    # Convert to TorchStructure (single frame)
    t = a.to_TorchStructure(frame=0)
    assert isinstance(t, TorchStructure)
    assert_array_equal(t.positions, coords)
    assert t.species == types
    assert t.energy == energy
    assert_array_equal(t.forces, forces)
    assert_array_equal(t.cell, cell)
    assert t.pbc is not None and len(t.pbc) == 3 and bool(np.all(t.pbc))

    # Back to AtomicStructure
    a2 = AtomicStructure.from_TorchStructure(t)
    assert a2.pbc
    assert_array_equal(a2.avec[-1], cell)
    assert_array_equal(a2.coords[-1], coords)
    # In AtomicStructure, forces may be stored as array in list
    f2 = a2.forces[-1]
    assert f2 is not None and len(f2) != 0
    assert_array_equal(f2, forces)
    assert a2.energy[-1] == energy


def test_isolated_no_forces_conversion():
    coords = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0]])
    types = ["Na", "Cl"]

    a = AtomicStructure(coords=coords, types=types, energy=None, forces=None)

    # Default returns list (even if single frame)
    traj = a.to_TorchStructure()
    assert isinstance(traj, list)
    assert len(traj) == 1
    t = traj[0]
    assert isinstance(t, TorchStructure)
    assert_array_equal(t.positions, coords)
    assert t.species == types
    assert t.energy is None
    assert t.forces is None
    assert t.cell is None
    assert t.pbc is None

    # Back to AtomicStructure
    a2 = AtomicStructure.from_TorchStructure(t)
    assert not a2.pbc
    assert a2.avec is None
    # forces stored as empty/None
    f2 = a2.forces[-1]
    assert (f2 is None) or (len(f2) == 0)
    assert a2.energy[-1] is None
    assert_array_equal(a2.coords[-1], coords)
    assert list(a2.types) == types


def test_multiframe_trajectory_handling_and_factory_alignment():
    # Build a two-frame periodic trajectory
    cell = np.eye(3) * 5.0
    coords0 = np.array([[0.0, 0.0, 0.0],
                        [0.5, 0.5, 0.5]])
    coords1 = coords0 + 0.1
    types = ["C", "C"]
    a = AtomicStructure(coords=coords0, types=types, avec=cell,
                        fractional=False, energy=0.0, forces=None)
    a.add_frame(coords1, avec=cell, energy=1.0, forces=None)

    # .to_TorchStructure() returns list of per-frame TorchStructure
    traj = a.to_TorchStructure()
    assert isinstance(traj, list)
    assert len(traj) == 2
    assert isinstance(traj[0], TorchStructure)
    assert isinstance(traj[1], TorchStructure)
    assert_array_equal(traj[0].positions, coords0)
    assert_array_equal(traj[1].positions, coords1)
    assert traj[0].energy == 0.0
    assert traj[1].energy == 1.0

    # Specific frame selection
    last = a.to_TorchStructure(frame=-1)
    first = a.to_TorchStructure(frame=0)
    assert_array_equal(last.positions, coords1)
    assert_array_equal(first.positions, coords0)

    # Factory method on Torch side should match selection
    t_last = TorchStructure.from_AtomicStructure(a, frame=-1)
    t_first = TorchStructure.from_AtomicStructure(a, frame=0)
    assert_array_equal(t_last.positions, coords1)
    assert_array_equal(t_first.positions, coords0)
    assert t_last.energy == 1.0
    assert t_first.energy == 0.0


def test_species_length_validation_in_torch_structure():
    # Ensure TorchStructure validation triggers if species length mismatches
    positions = np.zeros((3, 3))
    species = ["H", "H"]  # too short
    with pytest.raises(ValueError):
        TorchStructure(positions=positions, species=species, energy=0.0)
