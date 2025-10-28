import json
import csv
from pathlib import Path

import torch
import pytest

from aenet.torch_training import (
    TorchANNPotential,
    save_model,
    load_model,
    export_history,
)
from aenet.torch_featurize import ChebyshevDescriptor


@pytest.mark.cpu
def test_save_and_load_roundtrip(tmp_path: Path):
    # Minimal descriptor and architecture
    species = ["H"]
    descriptor = ChebyshevDescriptor(
        species=species,
        rad_order=0,
        rad_cutoff=3.0,
        ang_order=0,
        ang_cutoff=3.0,
        min_cutoff=0.55,
        device="cpu",
        dtype=torch.float64,
    )
    arch = {"H": [(2, "tanh")]}  # one hidden layer, 2 nodes

    # Trainer (untrained is fine for round-trip)
    trainer = TorchANNPotential(arch=arch, descriptor=descriptor)

    # Save
    out_path = tmp_path / "model_roundtrip.pt"
    save_model(trainer, out_path)

    assert out_path.exists(), "Model file was not created."

    # Load
    loaded_trainer, metadata = load_model(out_path)

    # Verify architecture equivalence
    assert loaded_trainer.arch == trainer.arch

    # Verify descriptor attributes
    ld = loaded_trainer.descriptor
    td = trainer.descriptor
    assert list(ld.species) == list(td.species)
    assert ld.rad_order == td.rad_order
    assert ld.ang_order == td.ang_order
    assert pytest.approx(ld.rad_cutoff) == td.rad_cutoff
    assert pytest.approx(ld.ang_cutoff) == td.ang_cutoff
    assert pytest.approx(ld.min_cutoff) == td.min_cutoff
    # dtype/device stored as strings; trainer stores in descriptor
    assert str(ld.dtype) == str(td.dtype)

    # Verify model parameters identical
    state1 = {k: v.detach().cpu()
              for k, v in trainer.model.state_dict().items()}
    state2 = {k: v.detach().cpu()
              for k, v in loaded_trainer.model.state_dict().items()}
    assert state1.keys() == state2.keys()
    for k in state1:
        if torch.is_floating_point(state1[k]):
            assert torch.allclose(state1[k], state2[k], atol=0.0, rtol=0.0)
        else:
            assert torch.equal(state1[k], state2[k])

    # Metadata smoke checks
    assert isinstance(metadata, dict)
    assert "descriptor_config" in metadata
    assert "architecture" in metadata
    assert "schema_version" in metadata


@pytest.mark.cpu
def test_export_history_json_and_csv(tmp_path: Path):
    history = {
        "train_energy_rmse": [1.0, 0.8, 0.6],
        "test_energy_rmse": [1.1, 0.9, 0.7],
        "train_force_rmse": [float("nan"), 0.5, 0.4],
        "test_force_rmse": [float("nan"), 0.6, 0.5],
        "learning_rates": [1e-3, 1e-3, 5e-4],
        "epoch_times": [0.1, 0.1, 0.1],
    }

    json_path = tmp_path / "history.json"
    csv_path = tmp_path / "history.csv"

    export_history(history, json_path, csv_path)

    # JSON checks
    assert json_path.exists(), "JSON history file not created."
    with open(json_path, "r", encoding="utf-8") as f:
        hist_json = json.load(f)
    for key in history.keys():
        assert key in hist_json

    # CSV checks
    assert csv_path.exists(), "CSV history file not created."
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # header + 3 epochs
    assert len(rows) == 4
    header = rows[0]
    assert header[:2] == ["epoch", "train_energy_rmse"]
