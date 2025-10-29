#!/usr/bin/env python
"""
Unified profiling for aenet.torch_training (CPU).

Modes:
  - features: micro-benchmarks of ChebyshevDescriptor features/gradients
  - training: end-to-end training profiling on XSF data (TiO2)
  - both: run features then training

Examples
--------
  # Feature-only (synthetic supercells)
  conda run -n aenet-torch python scripts/profile_training.py \
    --mode features --sizes 2 4 6 --repeats 3

  # Training profiling on TiO2 data (on-the-fly featurization)
  conda run -n aenet-torch python scripts/profile_training.py --mode training \
    --xsf-dir notebooks/xsf-TiO2 --limit 32 --epochs 5 --batch-size 32 \
    --force-weight 0.5 --profile-level detailed \
    --output outputs/profile_tio2_cpu.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from aenet.formats.xsf import XSFParser

# Local imports from this repository
from aenet.torch_featurize.featurize import ChebyshevDescriptor
from aenet.torch_training.config import Adam, Structure, TorchTrainingConfig
from aenet.torch_training.trainer import TorchANNPotential

# ------------- Utilities -------------


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "num_threads": torch.get_num_threads(),
        "device_available_cuda": torch.cuda.is_available(),
        "timestamp": now_iso(),
    }
    try:
        import platform

        info.update(
            {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "machine": platform.machine(),
                "python_implementation": platform.python_implementation(),
            }
        )
    except Exception:
        pass
    return info


def resolve_dtype_for_device(device: str, precision: str) -> torch.dtype:
    if precision == "float32":
        return torch.float32
    if precision == "float64":
        return torch.float64
    # auto: default to fp32 on CPU; keep fp32 for CUDA here (GPU tuning later)
    return torch.float32

# ------------- Feature/Gradient Micro-benchmarks -------------


def build_sc_cell(n: int, alat: float,
                  dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    coords = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                coords.append([i * alat, j * alat, k * alat])
    positions = torch.tensor(coords, dtype=dtype)
    cell = torch.eye(3, dtype=dtype) * (n * alat)
    return positions, cell


def jitter_positions(positions: torch.Tensor,
                     scale: float = 1e-3) -> torch.Tensor:
    if scale <= 0:
        return positions
    noise = (torch.rand_like(positions) - 0.5) * 2.0 * scale
    return positions + noise


def count_pairs_triplets(positions: torch.Tensor,
                         cell: torch.Tensor,
                         pbc: torch.Tensor,
                         descriptor: ChebyshevDescriptor) -> Tuple[int, int]:
    positions = positions.to(descriptor.dtype).to(descriptor.device)
    cell = cell.to(descriptor.dtype).to(descriptor.device)

    nbl = descriptor.nbl
    neighbor_data = nbl.get_neighbors(positions, cell, pbc, fractional=False)
    edge_index = neighbor_data["edge_index"]
    distances = neighbor_data["distances"]
    offsets = neighbor_data["offsets"]

    # Min cutoff filter matches featurizer
    mask = distances > descriptor.min_cutoff
    edge_index = edge_index[:, mask]
    distances = distances[mask]
    if offsets is not None:
        offsets = offsets[mask]

    # Pairs for radial: within radial cutoff
    pair_mask = distances <= descriptor.rad_cutoff
    n_pairs = int(pair_mask.sum().item())

    # Triplets per center: per-center neighbors within angular cutoff
    i_idx = edge_index[0]
    j_idx = edge_index[1]

    # Compute displacement vectors with offsets
    if cell is not None and offsets is not None:
        r_ij = (positions[j_idx] + offsets.to(descriptor.dtype)
                @ cell) - positions[i_idx]
    else:
        r_ij = positions[j_idx] - positions[i_idx]
    d_ij = torch.norm(r_ij, dim=-1)

    ang_mask = (d_ij <= descriptor.ang_cutoff) & (d_ij > descriptor.min_cutoff)
    i_idx_ang = i_idx[ang_mask]

    # Count combinations per center: C(m,2)
    n_triplets = 0
    if len(i_idx_ang) > 0:
        max_i = int(i_idx_ang.max().item()) + 1
        counts = torch.bincount(
            i_idx_ang, minlength=max(positions.shape[0], max_i))
        n_triplets = int(torch.sum(counts * (counts - 1) // 2).item())

    return n_pairs, n_triplets


def profile_features_once(n: int, args: argparse.Namespace) -> Dict[str, Any]:
    device = "cpu"  # CPU only per current scope
    precision = getattr(args, "precision", "auto")
    dtype = resolve_dtype_for_device(device, precision)

    alat = args.alat
    positions, cell = build_sc_cell(n, alat, dtype)
    positions = jitter_positions(positions, scale=args.jitter)
    pbc = torch.tensor([True, True, True])

    # Descriptor configuration
    species_list: List[str] = ["H"] if args.single_species else ["A", "B"]
    species = (["H"] * len(positions)
               if args.single_species else ["A"] * len(positions))

    descriptor = ChebyshevDescriptor(
        species=species_list,
        rad_order=args.rad_order,
        rad_cutoff=args.rad_cutoff,
        ang_order=args.ang_order,
        ang_cutoff=args.ang_cutoff,
        device=device,
        dtype=dtype,
    )

    # Warmup
    _ = descriptor.forward_from_positions(positions, species, cell, pbc)

    # Count neighbors
    n_pairs, n_triplets = count_pairs_triplets(
        positions, cell, pbc, descriptor)

    # Profile features
    feat_times: List[float] = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        _ = descriptor.forward_from_positions(positions, species, cell, pbc)
        t1 = time.perf_counter()
        feat_times.append(t1 - t0)

    # Profile gradients
    grad_times: List[float] = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        _ = descriptor.compute_feature_gradients(positions, species, cell, pbc)
        t1 = time.perf_counter()
        grad_times.append(t1 - t0)

    return {
        "n": n,
        "atoms": positions.shape[0],
        "dtype": str(dtype).replace("torch.", ""),
        "pairs": n_pairs,
        "triplets": n_triplets,
        "feat_mean": float(np.mean(feat_times)),
        "feat_min": float(np.min(feat_times)),
        "feat_max": float(np.max(feat_times)),
        "grad_mean": float(np.mean(grad_times)),
        "grad_min": float(np.min(grad_times)),
        "grad_max": float(np.max(grad_times)),
    }


def run_features_mode(args: argparse.Namespace) -> Dict[str, Any]:
    print("Feature/gradient profiling (CPU)")
    print(f"  sizes         : {args.sizes}")
    print(f"  alat          : {args.alat}")
    print(f"  rad_cutoff    : {args.rad_cutoff}")
    print(f"  ang_cutoff    : {args.ang_cutoff}")
    print(f"  rad_order     : {args.rad_order}")
    print(f"  ang_order     : {args.ang_order}")
    print(f"  repeats       : {args.repeats}")
    print(f"  jitter        : {args.jitter}")
    print(f"  single_species: {args.single_species}")
    print("")

    header = (
        "N  atoms    pairs      triplets    feat_mean(s)  grad_mean(s)  "
        "feat[min,max]  grad[min,max]"
    )
    print(header)
    print("-" * len(header))

    results: List[Dict[str, Any]] = []
    for n in args.sizes:
        res = profile_features_once(n, args)
        results.append(res)
        print(
            f"{res['n']:>1}  {res['atoms']:>5}  "
            f"{res['pairs']:>9}  {res['triplets']:>11}  "
            f"{res['feat_mean']:>12.6f}  {res['grad_mean']:>12.6f}  "
            f"[{res['feat_min']:.6f},{res['feat_max']:.6f}]  "
            f"[{res['grad_min']:.6f},{res['grad_max']:.6f}]"
        )
    return {"feature_benchmarks": results}


# ------------- Training Profiling -------------

def run_dataload_mode(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Pure DataLoader iteration timing to isolate featurization+collation cost.

    Iterates over StructureDataset with the given DataLoader worker settings
    and reports total iteration time, batches, mean batch time, and atoms/sec.
    """
    # Reduce potential thread oversubscription interference
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    xsf_dir = Path(args.xsf_dir)
    structures = load_xsf_structures(xsf_dir, limit=args.limit)

    # Resolve dtype
    dtype = resolve_dtype_for_device("cpu", getattr(args, "precision", "auto"))

    # Descriptor parameters (reuse feature-mode params)
    species = species_from_structures(structures)
    descriptor = ChebyshevDescriptor(
        species=species,
        rad_order=args.rad_order,
        rad_cutoff=args.rad_cutoff,
        ang_order=args.ang_order,
        ang_cutoff=args.ang_cutoff,
        device="cpu",
        dtype=dtype,
    )

    # Dataset and DataLoader
    from torch.utils.data import DataLoader as _DL

    from aenet.torch_training.dataset import StructureDataset as _DS
    from aenet.torch_training.trainer import _collate_fn as _collate

    ds = _DS(
        structures=structures,
        descriptor=descriptor,
        force_fraction=1.0,
        force_sampling="random",
        max_energy=None,
        max_forces=None,
        seed=args.seed,
    )

    dl_kwargs: Dict[str, Any] = {}
    nw = int(getattr(args, "num_workers", 0))
    if nw > 0:
        dl_kwargs.update(
            dict(
                num_workers=nw,
                prefetch_factor=int(getattr(args, "prefetch_factor", 2)),
                persistent_workers=bool(
                    getattr(args, "persistent_workers", "on") == "on"),
            )
        )
    else:
        dl_kwargs.update(dict(num_workers=0))

    loader = _DL(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=_collate,
        **dl_kwargs,
    )

    # Optional warmup for initial worker spin-up
    try:
        _ = next(iter(loader))
    except StopIteration:
        pass
    except Exception:
        # Ignore warmup errors; proceed to timed pass
        pass

    # Timed full pass
    batches = 0
    total_atoms = 0
    batch_times: List[float] = []
    t0 = time.perf_counter()
    for b in loader:
        t_b0 = time.perf_counter()
        # Count atoms in this batch (sum over structures in batch)
        try:
            total_atoms += int(b["n_atoms"].sum().item())
        except Exception:
            pass
        batches += 1
        batch_times.append(time.perf_counter() - t_b0)
    total_iter_time = time.perf_counter() - t0
    mean_batch_time = (float(np.mean(batch_times))
                       if len(batch_times) > 0 else float("nan"))
    atoms_per_sec = (float(total_atoms) / total_iter_time
                     ) if total_iter_time > 0.0 else float("nan")

    summary = {
        "dataset": {
            "n_structures": len(structures),
            "avg_atoms_per_structure": float(
                np.mean([len(s.positions) for s in structures])
                ) if structures else 0.0,
            "total_atoms": int(sum(len(s.positions) for s in structures)),
            "species": species,
            "xsf_dir": str(xsf_dir),
        },
        "descriptor": {
            "rad_order": args.rad_order,
            "rad_cutoff": args.rad_cutoff,
            "ang_order": args.ang_order,
            "ang_cutoff": args.ang_cutoff,
            "n_features": int(descriptor.get_n_features()),
            "dtype": str(descriptor.dtype).replace("torch.", ""),
        },
        "loader": {
            "batch_size": int(args.batch_size),
            "num_workers": nw,
            "prefetch_factor": int(getattr(args, "prefetch_factor", 2)),
            "persistent_workers": bool(
                getattr(args, "persistent_workers", "on") == "on"),
        },
        "timing": {
            "total_iter_time": total_iter_time,
            "batches": batches,
            "mean_batch_time": mean_batch_time,
            "atoms_per_sec": atoms_per_sec,
        },
    }

    print("")
    print("DataLoader Featurization Benchmark (CPU)")
    print("=======================================")
    print(f"Structures      : {summary['dataset']['n_structures']}")
    print("Avg atoms/struct: "
          + f"{summary['dataset']['avg_atoms_per_structure']:.1f}")
    print(f"Total atoms     : {summary['dataset']['total_atoms']}")
    print(f"Species         : {summary['dataset']['species']}")
    print(f"Batch size      : {summary['loader']['batch_size']}")
    print(f"Workers         : {summary['loader']['num_workers']}")
    print(f"Prefetch        : {summary['loader']['prefetch_factor']}")
    print(f"Persistent      : {summary['loader']['persistent_workers']}")
    print(f"Total iter time : {summary['timing']['total_iter_time']:.3f}s")
    print(f"Batches         : {summary['timing']['batches']}")
    if not np.isnan(mean_batch_time):
        print(f"Mean batch time : {summary['timing']['mean_batch_time']:.4f}s")
    if not np.isnan(atoms_per_sec):
        print(f"Atoms/sec       : {summary['timing']['atoms_per_sec']:.1f}")

    return {"dataload_profile": summary}


def load_xsf_structures(xsf_dir: Path,
                        limit: Optional[int] = None) -> List[Structure]:
    files = sorted([p for p in xsf_dir.glob("*.xsf")])
    if limit is not None:
        files = files[: int(limit)]
    if len(files) == 0:
        raise FileNotFoundError(f"No .xsf files found in {xsf_dir}")

    parser = XSFParser()
    out: List[Structure] = []
    for p in files:
        s = parser.read(str(p))
        positions = np.array(s.coords[-1])
        species = list(s.types)
        # Energy
        if (getattr(s, "energy", None) is None
                or len(s.energy) == 0 or s.energy[-1] is None):
            raise RuntimeError(f"Structure {p.name} is not energy-labeled.")
        energy = float(s.energy[-1])
        # Forces (optional)
        if (getattr(s, "forces", None) is not None
                and len(s.forces) > 0 and s.forces[-1] is not None):
            forces = np.array(s.forces[-1])
        else:
            forces = None
        # Cell and PBC
        cell = np.array(s.avec[-1]) if getattr(s, "pbc", False) else None
        pbc = np.array([True, True, True]
                       ) if getattr(s, "pbc", False) else None

        out.append(
            Structure(
                positions=positions,
                species=species,
                energy=energy,
                forces=forces,
                cell=cell,
                pbc=pbc,
                name=p.name,
            )
        )
    return out


def species_from_structures(structures: List[Structure]) -> List[str]:
    sp: List[str] = []
    seen = set()
    for s in structures:
        for el in s.species:
            if el not in seen:
                seen.add(el)
                sp.append(el)
    return sp


def default_arch_for_species(
        species: List[str]) -> Dict[str, List[Tuple[int, str]]]:
    # Simple two-hidden-layer tanh architecture per species
    arch: Dict[str, List[Tuple[int, str]]] = {}
    for el in species:
        arch[el] = [(16, "tanh"), (16, "tanh")]
    return arch


def run_training_mode(args: argparse.Namespace) -> Dict[str, Any]:
    torch.set_num_threads(max(torch.get_num_threads(), 1))

    xsf_dir = Path(args.xsf_dir)
    structures = load_xsf_structures(xsf_dir, limit=args.limit)

    # Resolve dtype
    dtype = resolve_dtype_for_device("cpu", getattr(args, "precision", "auto"))

    # Descriptor parameters
    species = species_from_structures(structures)
    descriptor = ChebyshevDescriptor(
        species=species,
        rad_order=args.rad_order,
        rad_cutoff=args.rad_cutoff,
        ang_order=args.ang_order,
        ang_cutoff=args.ang_cutoff,
        device="cpu",
        dtype=dtype,
    )

    # Model architecture
    arch = default_arch_for_species(species)

    # Potential/trainer
    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    # Training config (CPU, no scheduler; energy_target=total
    # to avoid atomic ref issues)
    method = Adam(mu=args.lr, batchsize=args.batch_size)
    cfg = TorchTrainingConfig(
        iterations=args.epochs,
        method=method,
        testpercent=args.testpercent,
        force_weight=args.force_weight,
        force_fraction=float(args.force_fraction),
        force_sampling=str(args.force_sampling),
        force_resample_each_epoch=(args.force_resample_each_epoch == "on"),
        force_min_structures_per_epoch=int(
            args.force_min_structures_per_epoch),
        force_scale_unbiased=(args.force_scale_unbiased == "on"),
        cached_features_for_force=(args.cached_features_for_force == "on"),
        cache_neighbors=(args.cache_neighbors == "on"),
        cache_triplets=(args.cache_triplets == "on"),
        cache_persist_dir=(args.cache_persist_dir if args.cache_persist_dir else None),
        epochs_per_force_window=int(args.epochs_per_force_window),
        memory_mode="cpu",
        max_energy=None,
        max_forces=None,
        save_energies=False,
        save_forces=False,
        timing=False,
        device="cpu",
        precision=args.precision,
        energy_target="total",  # use total energies as labels
        E_atomic=None,
        normalize_features=True,
        normalize_energy=True,
        cached_features=(args.cached == "on"),
        # DataLoader worker controls
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=(args.persistent_workers == "on"),
        show_progress=False,
        show_batch_progress=False,
    )

    # High-level wall-time per epoch (derived from pot.history)
    t0 = time.perf_counter()
    history = pot.train(
        structures=structures,
        config=cfg,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=0,
        resume_from=None,
        save_best=False,
        use_scheduler=False,
    )
    total_time = time.perf_counter() - t0

    # Summaries
    n_epochs = len(history.get("epoch_times", []))
    mean_epoch_time = float(np.mean(history["epoch_times"])
                            ) if n_epochs > 0 else float("nan")
    mean_forward_time = float(np.mean(history["epoch_forward_time"])
                              ) if n_epochs > 0 else float("nan")
    mean_backward_time = float(np.mean(history["epoch_backward_time"])
                               ) if n_epochs > 0 else float("nan")
    # New detailed timings (available when trainer instrumentation is present)
    # Training split
    mean_data_loading_time_train = (
        float(np.mean(history["epoch_data_loading_time_train"]))
        if ("epoch_data_loading_time_train" in history
            and len(history["epoch_data_loading_time_train"]) > 0)
        else float("nan")
    )
    mean_loss_time_train = (
        float(np.mean(history["epoch_loss_time_train"]))
        if ("epoch_loss_time_train" in history
            and len(history["epoch_loss_time_train"]) > 0)
        else float("nan")
    )
    mean_optimizer_time_train = (
        float(np.mean(history["epoch_optimizer_time_train"]))
        if ("epoch_optimizer_time_train" in history
            and len(history["epoch_optimizer_time_train"]) > 0)
        else float("nan")
    )
    mean_train_time = (
        float(np.mean(history["epoch_train_time"]))
        if ("epoch_train_time" in history
            and len(history["epoch_train_time"]) > 0)
        else float("nan")
    )
    # Validation split
    mean_data_loading_time_val = (
        float(np.mean(history["epoch_data_loading_time_val"]))
        if ("epoch_data_loading_time_val" in history
            and len(history["epoch_data_loading_time_val"]) > 0)
        else float("nan")
    )
    mean_loss_time_val = (
        float(np.mean(history["epoch_loss_time_val"]))
        if ("epoch_loss_time_val" in history
            and len(history["epoch_loss_time_val"]) > 0)
        else float("nan")
    )
    mean_optimizer_time_val = (
        float(np.mean(history["epoch_optimizer_time_val"]))
        if ("epoch_optimizer_time_val" in history
            and len(history["epoch_optimizer_time_val"]) > 0)
        else float("nan")
    )
    mean_val_time = (
        float(np.mean(history["epoch_val_time"]))
        if "epoch_val_time" in history and len(history["epoch_val_time"]) > 0
        else float("nan")
    )
    # Totals across splits (when available)
    mean_data_loading_time_total = (
        (0.0 if np.isnan(mean_data_loading_time_train)
         else mean_data_loading_time_train)
        + (0.0 if np.isnan(mean_data_loading_time_val)
           else mean_data_loading_time_val)
    )
    mean_loss_time_total = (
        (0.0 if np.isnan(mean_loss_time_train) else mean_loss_time_train)
        + (0.0 if np.isnan(mean_loss_time_val) else mean_loss_time_val)
    )
    mean_optimizer_time_total = (
        (0.0 if np.isnan(mean_optimizer_time_train)
         else mean_optimizer_time_train)
        + (0.0 if np.isnan(mean_optimizer_time_val)
           else mean_optimizer_time_val)
    )
    featurization_fraction = (
        (mean_data_loading_time_total / mean_epoch_time)
        if (not np.isnan(mean_epoch_time) and mean_epoch_time > 0.0)
        else None
    )

    summary = {
        "dataset": {
            "n_structures": len(structures),
            "avg_atoms_per_structure": float(
                np.mean([len(s.positions) for s in structures])
            ),
            "species": species,
            "xsf_dir": str(xsf_dir),
        },
        "descriptor": {
            "rad_order": args.rad_order,
            "rad_cutoff": args.rad_cutoff,
            "ang_order": args.ang_order,
            "ang_cutoff": args.ang_cutoff,
            "n_features": int(descriptor.get_n_features()),
            "dtype": str(descriptor.dtype).replace("torch.", ""),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "force_weight": args.force_weight,
            "force_fraction": args.force_fraction,
            "force_sampling": args.force_sampling,
            "force_resample_each_epoch": args.force_resample_each_epoch,
            "force_min_structures_per_epoch": (
                args.force_min_structures_per_epoch),
            "force_scale_unbiased": args.force_scale_unbiased,
            "cached_features_for_force": args.cached_features_for_force,
            "cache_neighbors": args.cache_neighbors,
            "cache_triplets": args.cache_triplets,
            "cache_persist_dir": args.cache_persist_dir,
            "epochs_per_force_window": args.epochs_per_force_window,
            "testpercent": args.testpercent,
            "lr": args.lr,
            "arch_hidden": arch,
        },
        "timing": {
            "total_time": total_time,
            "mean_epoch_time": mean_epoch_time,
            "mean_forward_time": mean_forward_time,
            "mean_backward_time": mean_backward_time,
            # Train split means
            "mean_data_loading_time_train": mean_data_loading_time_train,
            "mean_loss_time_train": mean_loss_time_train,
            "mean_optimizer_time_train": mean_optimizer_time_train,
            "mean_train_time": mean_train_time,
            # Val split means
            "mean_data_loading_time_val": mean_data_loading_time_val,
            "mean_loss_time_val": mean_loss_time_val,
            "mean_optimizer_time_val": mean_optimizer_time_val,
            "mean_val_time": mean_val_time,
            # Totals and fractions
            "mean_data_loading_time_total": mean_data_loading_time_total,
            "mean_loss_time_total": mean_loss_time_total,
            "mean_optimizer_time_total": mean_optimizer_time_total,
            "featurization_fraction": featurization_fraction,
            # Per-epoch raw arrays
            "epoch_times": history.get("epoch_times", []),
            "epoch_forward_time": history.get("epoch_forward_time", []),
            "epoch_backward_time": history.get("epoch_backward_time", []),
            "epoch_data_loading_time_train": history.get(
                "epoch_data_loading_time_train", []),
            "epoch_loss_time_train": history.get("epoch_loss_time_train", []),
            "epoch_optimizer_time_train": history.get(
                "epoch_optimizer_time_train", []),
            "epoch_data_loading_time_val": history.get(
                "epoch_data_loading_time_val", []),
            "epoch_loss_time_val": history.get(
                "epoch_loss_time_val", []),
            "epoch_optimizer_time_val": history.get(
                "epoch_optimizer_time_val", []),
            "epoch_train_time": history.get("epoch_train_time", []),
            "epoch_val_time": history.get("epoch_val_time", []),
        },
        "history_tail": {
            "train_energy_rmse_last": (
                history["train_energy_rmse"][-1]
                if history["train_energy_rmse"] else None),
            "train_force_rmse_last": (
                history["train_force_rmse"][-1]
                if history["train_force_rmse"] else None),
            "test_energy_rmse_last": (
                history["test_energy_rmse"][-1]
                if history["test_energy_rmse"] else None),
            "test_force_rmse_last": (
                history["test_force_rmse"][-1]
                if history["test_force_rmse"] else None),
        },
    }

    # Console output
    print("")
    print("Training Profiling Summary (CPU)")
    print("================================")
    print(f"Structures      : {summary['dataset']['n_structures']}")
    print(f"Species         : {summary['dataset']['species']}")
    print(f"Epochs          : {args.epochs}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Force weight    : {args.force_weight}")
    print(f"Descriptor F    : {summary['descriptor']['n_features']}")
    print(f"Total time      : {summary['timing']['total_time']:.3f}s")
    if not np.isnan(mean_epoch_time):
        print(f"Mean epoch time : {summary['timing']['mean_epoch_time']:.3f}s")
        print(f"  - forward     : {summary['timing']['mean_forward_time']:.3f}s")
        print(f"  - backward    : {summary['timing']['mean_backward_time']:.3f}s")
        # Train split breakdown
        if not np.isnan(mean_train_time):
            print(f"Train split     : {summary['timing']['mean_train_time']:.3f}s")
        if not np.isnan(mean_data_loading_time_train):
            print(f"  - data_load   : {summary['timing']['mean_data_loading_time_train']:.3f}s")
        if not np.isnan(mean_loss_time_train):
            print(f"  - loss        : {summary['timing']['mean_loss_time_train']:.3f}s")
        if not np.isnan(mean_optimizer_time_train):
            print(f"  - optimizer   : {summary['timing']['mean_optimizer_time_train']:.3f}s")
        # Validation split breakdown
        if not np.isnan(mean_val_time):
            print(f"Val split       : {summary['timing']['mean_val_time']:.3f}s")
        if not np.isnan(mean_data_loading_time_val):
            print(f"  - data_load   : {summary['timing']['mean_data_loading_time_val']:.3f}s")
        if not np.isnan(mean_loss_time_val):
            print(f"  - loss        : {summary['timing']['mean_loss_time_val']:.3f}s")
        if not np.isnan(mean_optimizer_time_val):
            print(f"  - optimizer   : {summary['timing']['mean_optimizer_time_val']:.3f}s")
        # Totals
        if featurization_fraction is not None:
            print(f"Featurization fraction (train+val data_load)/epoch: {featurization_fraction*100.0:.1f}%")

    return {"training_profile": summary, "history": history}


# ------------- Main CLI -------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "profile_training.py"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["features", "training", "both", "dataload"],
        default="training",
        help="Which profiling mode to run.",
    )
    # Feature mode args
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 6], help="Supercell edges N for NxNxN atoms.")
    parser.add_argument("--alat", type=float, default=2.5, help="Lattice parameter (Angstrom).")
    parser.add_argument("--rad-cutoff", type=float, default=4.0, dest="rad_cutoff", help="Radial cutoff radius.")
    parser.add_argument("--ang-cutoff", type=float, default=3.0, dest="ang_cutoff", help="Angular cutoff radius.")
    parser.add_argument("--rad-order", type=int, default=5, dest="rad_order")
    parser.add_argument("--ang-order", type=int, default=3, dest="ang_order")
    parser.add_argument("--repeats", type=int, default=3, help="Timing repeats per size.")
    parser.add_argument("--jitter", type=float, default=1e-3, help="Random displacement scale.")
    parser.add_argument("--single-species", action="store_true", help="Use single-species mode for features bench.")

    # Training mode args
    parser.add_argument("--xsf-dir", type=str, default="notebooks/xsf-TiO2", help="Directory with .xsf structures.")
    parser.add_argument("--limit", type=int, default=32, help="Limit number of .xsf structures to load.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs to run.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for Adam optimizer.")
    parser.add_argument("--force-weight", type=float, default=0.0, help="alpha: weight for force loss [0..1].")
    parser.add_argument("--testpercent", type=int, default=10, help="Validation split percentage [0..100].")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument(
        "--profile-level",
        type=str,
        choices=["quick", "detailed", "deep"],
        default="detailed",
        help="Profiling depth (currently informational; deep may enable cProfile in future).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON profiling output. If not provided, only prints to console.",
    )
    parser.add_argument(
        "--cached",
        type=str,
        choices=["off", "on"],
        default="off",
        help="Use cached-features mode for energy-only training (precompute once).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["auto", "float32", "float64"],
        default="auto",
        help="Numeric precision (default auto uses float32 on CPU).",
    )
    # DataLoader worker controls (Issue 2)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for on-the-fly featurization.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch_factor when num_workers>0.")
    parser.add_argument(
        "--persistent-workers",
        type=str,
        choices=["off", "on"],
        default="on",
        help="Enable persistent_workers in DataLoader when num_workers>0.",
    )
    # Force subsampling controls (Issue 4)
    parser.add_argument(
        "--force-fraction",
        type=float,
        default=1.0,
        help="Fraction of force-labeled structures supervised per epoch [0..1].",
    )
    parser.add_argument(
        "--force-sampling",
        type=str,
        choices=["random", "fixed"],
        default="random",
        help="Force structure sampling mode.",
    )
    parser.add_argument(
        "--force-resample-each-epoch",
        type=str,
        choices=["off", "on"],
        default="on",
        help="Resample force structures every epoch when force_sampling=random.",
    )
    parser.add_argument(
        "--force-min-structures-per-epoch",
        type=int,
        default=1,
        help="Minimum number of force-labeled structures per epoch.",
    )
    parser.add_argument(
        "--force-scale-unbiased",
        type=str,
        choices=["off", "on"],
        default="off",
        help="Apply sqrt(1/f) scaling to force RMSE when sub-sampling forces.",
    )
    # Mixed-run feature caching for non-force structures
    parser.add_argument(
        "--cached-features-for-force",
        type=str,
        choices=["off", "on"],
        default="off",
        help="Cache features for structures not selected for forces in current window.",
    )
    parser.add_argument(
        "--epochs-per-force-window",
        type=int,
        default=1,
        help="Resample random force subset every this many epochs (>=1).",
    )
    # Neighbor caching (Issue 5)
    parser.add_argument(
        "--cache-neighbors",
        type=str,
        choices=["off", "on"],
        default="off",
        help="Cache per-structure neighbor graphs across epochs to reduce data loading.",
    )
    # Triplet/CSR caching and persistence (Issue 5 Phase 2 / Issue 7)
    parser.add_argument(
        "--cache-triplets",
        type=str,
        choices=["off", "on"],
        default="off",
        help="Build and cache CSR + Triplet indices per structure and use vectorized paths.",
    )
    parser.add_argument(
        "--cache-persist-dir",
        type=str,
        default=None,
        help="Optional directory to persist cached CSR/Triplets to disk (future).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    # Seed for reproducibility (affects sampling elsewhere)
    try:
        import random

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    except Exception:
        pass

    results: Dict[str, Any] = {"system": system_info(), "args": vars(args)}
    try:
        if args.mode in ("features", "both"):
            feat_res = run_features_mode(args)
            results.update(feat_res)

        if args.mode == "dataload":
            data_res = run_dataload_mode(args)
            results.update(data_res)

        if args.mode in ("training", "both"):
            train_res = run_training_mode(args)
            results.update(train_res)

    except Exception as e:
        print("[ERROR] Profiling failed:", e)
        traceback.print_exc()
        results["error"] = str(e)

    # Write JSON output if requested
    if args.output:
        out_path = Path(args.output)
        ensure_dir(out_path)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote profiling JSON to: {out_path}")

    # Also return non-zero if there was an error (useful for CI)
    if "error" in results:
        sys.exit(1)


if __name__ == "__main__":
    main()
