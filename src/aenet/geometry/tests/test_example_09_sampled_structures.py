import hashlib
import importlib.util
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
NOTEBOOK_PATH = (
    REPO_ROOT / "notebooks" / "example-09-sampled-structures-downselection.ipynb"
)
DATA_DIR = REPO_ROOT / "notebooks" / "data" / "NaCl-sampled-structures"
CI_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _notebook_source() -> tuple[dict, str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    return notebook, source


def _load_script(name: str, monkeypatch):
    scripts_dir = DATA_DIR / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    module_path = scripts_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(
        f"example_09_{name}", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tracked_archive_matches_manifest():
    manifest = json.loads(
        (DATA_DIR / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    archive_path = DATA_DIR / manifest["archive"]
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    with tarfile.open(archive_path, mode="r:xz") as archive:
        xsf_names = [
            member.name
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith(".xsf")
        ]

    assert digest == manifest["archive_sha256"]
    assert len(xsf_names) == manifest["num_structures"] == 20000
    assert all(
        not PurePosixPath(name).is_absolute()
        and ".." not in PurePosixPath(name).parts
        for name in xsf_names
    )
    counts = Counter(Path(name).stem.rsplit("_", 1)[-1] for name in xsf_names)
    assert counts == Counter(manifest["structures_by_temperature"])
    default_counts = Counter(
        Path(name).stem.rsplit("_", 1)[-1] for name in sorted(xsf_names)[:100]
    )
    assert default_counts == Counter(
        {
            "550K": 25,
            "700K": 25,
            "850K": 25,
            "1000K": 25,
        }
    )


def test_precomputed_features_match_manifest_and_structure_archive():
    manifest = json.loads(
        (DATA_DIR / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    feature_manifest = manifest["precomputed_features"]
    feature_path = DATA_DIR / feature_manifest["archive"]
    digest = hashlib.sha256(feature_path.read_bytes()).hexdigest()

    with np.load(feature_path, allow_pickle=False) as data:
        assert set(data.files) == {"features", "paths", "source_indices"}
        features = data["features"]
        paths = data["paths"].astype(str)
        source_indices = data["source_indices"]

    with tarfile.open(DATA_DIR / manifest["archive"], mode="r:xz") as archive:
        xsf_names = sorted(
            Path(member.name).name
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith(".xsf")
        )

    assert digest == feature_manifest["archive_sha256"]
    assert list(features.shape) == feature_manifest["shape"]
    assert str(features.dtype) == feature_manifest["dtype"]
    assert np.isfinite(features).all()
    assert paths.tolist() == [f"sampled_structures/{name}" for name in xsf_names]
    assert all(
        not PurePosixPath(path).is_absolute()
        and ".." not in PurePosixPath(path).parts
        for path in paths
    )
    assert np.array_equal(source_indices, np.arange(manifest["num_structures"]))


def test_notebook_uses_only_tracked_or_explicit_feature_inputs():
    notebook, source = _notebook_source()

    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    execution_counts = [cell["execution_count"] for cell in code_cells]
    code_line_counts = [
        len("".join(cell["source"]).splitlines()) for cell in code_cells
    ]

    assert execution_counts == list(range(1, len(code_cells) + 1))
    assert any(cell.get("outputs") for cell in code_cells)
    assert len(code_cells) <= 8
    assert sum(code_line_counts) <= 120
    assert max(code_line_counts) <= 30
    assert "data/NaCl-sampled-structures" in source
    assert "sampled_structures.tar.xz" in source
    assert "sampled_structure_features.npz" in source
    assert "structure_generation/sampled_structures" not in source
    assert "down_selection/sampled_feature_outputs" not in source
    assert "FEATURE_FILE" not in source
    assert "ChebyshevDescriptor" not in source
    assert "HDF5StructureDataset" not in source
    assert "TemporaryDirectory" not in source
    assert "archive.extract(structure_labels[index]" in source
    assert 'example-09-outputs" / "selected-structures' in source
    assert "all(path.is_file() for path in selected_structure_paths)" in source
    assert "/burg-archive/" not in NOTEBOOK_PATH.read_text(encoding="utf-8")


def test_ci_notebook_matrix_references_tracked_notebooks():
    ci_source = CI_PATH.read_text(encoding="utf-8")
    notebook_paths = [
        line.split("notebook:", 1)[1].strip()
        for line in ci_source.splitlines()
        if line.lstrip().startswith("- notebook:")
    ]

    assert notebook_paths
    assert all((REPO_ROOT / path).is_file() for path in notebook_paths)
    assert (
        "notebooks/example-09-sampled-structures-downselection.ipynb"
        in notebook_paths
    )


def test_notebook_keeps_sampling_and_visualization_spaces_separate():
    _, source = _notebook_source()

    assert "representative_subset(\n    scaled_features" in source
    assert "PCA(n_components=2).fit_transform(scaled_features)" in source
    assert "TSNE" not in source
    assert 'Path(label).stem.rsplit("_", 1)[-1]' in source


def test_conversion_script_uses_archive_filename_contract(monkeypatch):
    module = _load_script("traj_to_xsf", monkeypatch)

    assert module.snapshot_name(1, "550") == "snapshot_0001_550K.xsf"
    assert module.snapshot_name(5000, "1000") == "snapshot_5000_1000K.xsf"


def test_generation_protocol_matches_manifest(monkeypatch):
    module = _load_script("uma_md", monkeypatch)
    manifest = json.loads(
        (DATA_DIR / "dataset_manifest.json").read_text(encoding="utf-8")
    )["generation"]

    assert module.timestep_fs == manifest["timestep_fs"]
    assert module.equilibration_ps == manifest["equilibration_ps"]
    assert module.production_ps == manifest["production_ps"]
    assert (
        module.production_frames
        == manifest["production_frames_per_temperature"]
    )


def test_uma_md_selects_the_only_vasp_file(tmp_path, monkeypatch):
    module = _load_script("uma_md", monkeypatch)
    structure_path = tmp_path / "NaCl.vasp"
    structure_path.touch()

    assert module.select_vasp_file(tmp_path) == structure_path


def test_uma_md_requires_explicit_choice_for_multiple_vasp_files(
    tmp_path,
    monkeypatch,
):
    module = _load_script("uma_md", monkeypatch)
    first = tmp_path / "a.vasp"
    second = tmp_path / "b.vasp"
    first.touch()
    second.touch()

    with pytest.raises(ValueError, match="multiple .vasp files"):
        module.select_vasp_file(tmp_path)

    assert module.select_vasp_file(tmp_path, Path("b.vasp")) == second


def test_uma_md_rejects_missing_vasp_file(tmp_path, monkeypatch):
    module = _load_script("uma_md", monkeypatch)

    with pytest.raises(FileNotFoundError, match="no .vasp file"):
        module.select_vasp_file(tmp_path)


def test_cutoff_analysis_balances_separated_shells(monkeypatch):
    import numpy as np

    module = _load_script("analyze_chebyshev_cutoffs", monkeypatch)
    lower_shell = np.array([1.0, 1.1, 1.2, 1.3])
    upper_shell = np.array([1.8, 1.9, 2.0, 2.1])

    cutoff, stats = module.classification_cutoff(lower_shell, upper_shell)

    assert 1.3 <= cutoff <= 1.8
    assert stats["lower_capture"] == 1.0
    assert stats["upper_leakage"] == 0.0
