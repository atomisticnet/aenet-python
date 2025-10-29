import math

import numpy as np
import torch
import pytest

from aenet.torch_training import (
    TorchTrainingConfig,
    Structure,
    TorchANNPotential,
)
from aenet.torch_featurize import ChebyshevDescriptor


def _make_simple_structures_H_two():
    # Two small H-only structures
    pos_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.0, 0.9, 0.0],
        ],
        dtype=np.float64,
    )
    pos_b = np.array(
        [
            [0.1, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    species = ["H", "H", "H"]
    E_a = 0.0
    E_b = 0.5
    F_a = np.zeros_like(pos_a)
    F_b = np.zeros_like(pos_b)

    sA = Structure(positions=pos_a, species=species, energy=E_a, forces=F_a)
    sB = Structure(positions=pos_b, species=species, energy=E_b, forces=F_b)
    return [sA, sB]


def _make_descriptor_H(dtype=torch.float64):
    return ChebyshevDescriptor(
        species=["H"],
        rad_order=1,
        rad_cutoff=2.0,
        ang_order=0,
        ang_cutoff=2.0,
        min_cutoff=0.1,
        device="cpu",
        dtype=dtype,
    )


def _make_arch_H(_descriptor: ChebyshevDescriptor):
    return {
        "H": [(4, "tanh")],
    }


class FakeTqdm:
    """
    Test double for tqdm that can emulate both styles:
      - tqdm(total=..., desc=...) for outer epoch bar
      - tqdm(iterable, total=..., desc=...) for inner batch bar
    """
    instances = []

    def __init__(self, iterable=None, total=None,
                 desc=None, ncols=None, leave=None):
        self.iterable = iterable
        self.total = total
        self.desc = desc
        self.ncols = ncols
        self.leave = leave
        self.n = 0
        self.closed = False
        self._postfix = {}
        FakeTqdm.instances.append(self)

    def set_postfix(self, pf):
        self._postfix = dict(pf)

    def update(self, n=1):
        self.n += int(n)

    def refresh(self):
        pass

    def close(self):
        self.closed = True

    def __iter__(self):
        if self.iterable is None:
            # Non-iterable style; nothing to iterate
            return iter(())
        for item in self.iterable:
            yield item


@pytest.mark.cpu
def test_progress_bars_enabled(monkeypatch):
    # Arrange
    structures = _make_simple_structures_H_two()
    descriptor = _make_descriptor_H(dtype=torch.float64)
    arch = _make_arch_H(descriptor)
    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    # Monkeypatch tqdm inside trainer to our fake
    from aenet.torch_training import trainer as trainer_mod
    FakeTqdm.instances = []
    monkeypatch.setattr(trainer_mod, "tqdm", FakeTqdm, raising=True)

    cfg = TorchTrainingConfig(
        iterations=3,
        testpercent=50,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        save_energies=False,
        show_progress=True,
        show_batch_progress=True,  # exercise inner bar as well
    )

    # Act
    history = pot.train(
        structures=structures,
        config=cfg,
        checkpoint_dir=None,
        checkpoint_interval=0,
        save_best=False,
        use_scheduler=False,
    )

    # Assert training completed
    assert len(history["train_energy_rmse"]) == 3
    assert not math.isnan(history["train_energy_rmse"][-1])

    # Assert outer progress bar updated once per epoch
    outers = [inst for inst in FakeTqdm.instances if inst.iterable is None]
    assert len(outers) == 1, "Expected a single outer epoch progress bar"
    assert outers[0].n == 3, (
        "Outer progress bar should be updated per epoch"
    )

    # Inner progress bar(s) should have been created for batches
    inners = [inst for inst in FakeTqdm.instances if inst.iterable is not None]
    assert len(inners) >= 1, "Expected at least one inner batch progress bar"


@pytest.mark.cpu
def test_progress_bars_disabled(monkeypatch):
    # Arrange
    structures = _make_simple_structures_H_two()
    descriptor = _make_descriptor_H(dtype=torch.float64)
    arch = _make_arch_H(descriptor)
    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    # Monkeypatch tqdm to a raising stub; it must not be called
    class RaisingTqdm:
        def __init__(self, *args, **kwargs):
            msg = (
                "tqdm should not have been invoked when show_progress=False"
            )
            raise AssertionError(msg)

    from aenet.torch_training import trainer as trainer_mod
    monkeypatch.setattr(trainer_mod, "tqdm", RaisingTqdm, raising=True)

    cfg = TorchTrainingConfig(
        iterations=2,
        testpercent=0,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        save_energies=False,
        show_progress=False,
        show_batch_progress=True,  # should be ignored when show_progress=False
    )

    # Act - should not use tqdm at all
    history = pot.train(
        structures=structures,
        config=cfg,
        checkpoint_dir=None,
        checkpoint_interval=0,
        save_best=False,
        use_scheduler=False,
    )

    # Assert training completed
    assert len(history["train_energy_rmse"]) == 2
    assert not math.isnan(history["train_energy_rmse"][-1])
