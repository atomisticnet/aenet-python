"""
ASCII export of TorchANNPotential to aenet '.nn.ascii' format.

This module adapts the writer logic from aenet-PyTorch (Jon Lopez-Zorilla)
to work with our TorchANNPotential + EnergyModelAdapter abstraction.

Primary entrypoint:
    export_to_ascii_impl(trainer, output_dir, prefix, ...)

Notes
-----
- Currently supports ChebyshevDescriptor.
- Descriptor statistics expected by aenet are:
    sfval_min, sfval_max, sfval_avg, sfval_cov (E[x^2], not variance).
- When 'structures' are provided, we compute exact min/max/avg/cov via a
  single pass over features; otherwise we derive avg/cov from the trainer's
  NormalizationManager if available and approximate min/max as avg +/- std.

File format
-----------
Per-species '.nn.ascii' files are written in the following order:
1) Network section  (weights, activations)
2) Setup  section   (descriptor configuration + feature statistics)
3) Trainset info    (normalization metadata + dataset summary)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import io
import numpy as np
import torch
import torch.nn as nn

PathLike = Union[str, Path]


@dataclass
class _DescriptorStats:
    """Container for descriptor feature statistics."""
    min: np.ndarray
    max: np.ndarray
    avg: np.ndarray
    cov: np.ndarray  # E[x^2], not variance


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _as_numpy_1d(x: Any) -> np.ndarray:
    if x is None:
        raise ValueError("Expected array-like, got None")
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1)
    arr = np.asarray(x)
    return arr.reshape(-1)


def _activation_code(name: str) -> int:
    """
    Map activation name to aenet integer code.
      linear -> 0
      tanh   -> 1
      sigmoid-> 2
    """
    n = name.lower()
    if n == "linear":
        return 0
    if n == "tanh":
        return 1
    if n == "sigmoid":
        return 2
    # Default to linear if unknown (conservative)
    return 0


def _extract_hidden_and_activations_from_seq(
        seq: nn.Sequential
        ) -> Tuple[List[int], List[str]]:
    """
    Infer hidden layer sizes and activations from a per-species Sequential.

    Returns
    -------
    hidden_sizes : List[int]
    activations  : List[str]
    """
    hidden: List[int] = []
    acts: List[str] = []
    linears: List[nn.Linear] = [m for m in seq if isinstance(m, nn.Linear)]
    if not linears:
        return hidden, acts
    # All but last Linear are hidden layers; last is output to 1
    for lin in linears[:-1]:
        hidden.append(int(lin.out_features))
    # Activations: scan modules in order; record activation after
    # each hidden Linear. We assume typical pattern
    # [Linear, Act, Linear, Act, ..., Linear(out)]
    act_names: List[str] = []
    for m in seq:
        if isinstance(m, nn.Tanh):
            act_names.append("tanh")
        elif isinstance(m, nn.Sigmoid):
            act_names.append("sigmoid")
        elif isinstance(m, nn.Identity):
            act_names.append("linear")
        else:
            raise NotImplementedError(
                f"Unsupported activation module type: {type(m)}")
    # Truncate to number of hidden layers
    if len(act_names) >= len(hidden):
        acts = act_names[: len(hidden)]
    else:
        # Pad with linear if missing activations
        acts = act_names + ["linear"] * (len(hidden) - len(act_names))
    return hidden, acts


def _get_species_list(trainer) -> List[str]:
    sp = list(getattr(trainer.descriptor, "species", []))
    return [str(s) for s in sp]


def _get_input_size(trainer) -> int:
    try:
        return int(trainer.descriptor.get_n_features())
    except Exception:
        # Fallback to first per-species Linear in functions
        seq0: nn.Sequential = trainer.net.functions[0]
        lin0 = next((m for m in seq0 if isinstance(m, nn.Linear)), None)
        if lin0 is None:
            raise RuntimeError("Cannot infer input feature size.")
        return int(lin0.in_features)


def _gather_descriptor_stats_from_structures(
    trainer,
    structures: Sequence[Any],
) -> _DescriptorStats:
    """
    Compute min/max/avg/cov for descriptor features by iterating structures.
    """
    desc = trainer.descriptor
    dtype = desc.dtype
    n_features = int(desc.get_n_features())
    min_v = np.full((n_features,), np.inf, dtype=np.float64)
    max_v = np.full((n_features,), -np.inf, dtype=np.float64)
    sum_v = np.zeros((n_features,), dtype=np.float64)
    sumsq_v = np.zeros((n_features,), dtype=np.float64)
    total = 0

    with torch.no_grad():
        for s in structures:
            positions = torch.as_tensor(s.positions, dtype=dtype)
            cell = torch.as_tensor(
                s.cell, dtype=dtype) if s.cell is not None else None
            pbc = torch.as_tensor(s.pbc) if s.pbc is not None else None
            feats = desc.forward_from_positions(
                positions, s.species, cell, pbc)  # (N, F)
            f = feats.detach().cpu().numpy().astype(np.float64, copy=False)
            min_v = np.minimum(min_v, f.min(axis=0))
            max_v = np.maximum(max_v, f.max(axis=0))
            sum_v += f.sum(axis=0)
            sumsq_v += (f * f).sum(axis=0)
            total += f.shape[0]

    if total == 0:
        # Degenerate: all zeros
        return _DescriptorStats(
            min=np.zeros(n_features, dtype=np.float64),
            max=np.zeros(n_features, dtype=np.float64),
            avg=np.zeros(n_features, dtype=np.float64),
            cov=np.zeros(n_features, dtype=np.float64),
        )

    avg = sum_v / float(total)
    ex2 = sumsq_v / float(total)  # E[x^2] (cov in aenet naming)
    return _DescriptorStats(min=min_v, max=max_v, avg=avg, cov=ex2)


def _derive_descriptor_stats_from_normalizer(
        trainer) -> Optional[_DescriptorStats]:
    """
    Build statistics using NormalizationManager.

    Prefer exact min/max if available; otherwise:
      avg = mean
      cov = std^2 + mean^2
      min/max ≈ avg ± std
    """
    try:
        norm = getattr(trainer, "_normalizer", None)
        if (norm is None or norm.feature_mean is None
                or norm.feature_std is None):
            return None
        mean = _as_numpy_1d(norm.feature_mean)
        std = _as_numpy_1d(norm.feature_std)
        cov = std * std + mean * mean
        # Use exact min/max if persisted, else approximate
        if (getattr(norm, "feature_min", None) is not None
                and getattr(norm, "feature_max", None) is not None):
            min_v = _as_numpy_1d(norm.feature_min)
            max_v = _as_numpy_1d(norm.feature_max)
        else:
            min_v = mean - std
            max_v = mean + std
        return _DescriptorStats(min=min_v, max=max_v, avg=mean, cov=cov)
    except Exception:
        return None


def _write_network_section(
    f: io.TextIOBase,
    trainer,
    species_index: int,
    species_name: str,
    input_size: int,
) -> None:
    """
    Write the network section for a single species to file-like 'f'.
    """
    # Resolve hidden sizes and activations
    hidden_sizes: List[int]
    activations: List[str]
    net = trainer.net
    seq: nn.Sequential = net.functions[species_index]

    # Prefer NetAtom introspection if available
    if hasattr(net, "hidden_size") and hasattr(net, "active_names"):
        try:
            hidden_sizes = [int(x) for x in net.hidden_size[species_index]]
            activations = [str(a).lower()
                           for a in net.active_names[species_index]]
        except Exception:
            (hidden_sizes, activations
             ) = _extract_hidden_and_activations_from_seq(seq)
    else:
        (hidden_sizes, activations
         ) = _extract_hidden_and_activations_from_seq(seq)

    # Nodes per layer (input, hidden..., output=1)
    nnodes = [input_size] + hidden_sizes + [1]
    nlayers = len(nnodes)
    nnodesmax = int(np.max(np.asarray(nnodes)))

    # Activation codes per layer transition; output layer is linear(0)
    fun = [_activation_code(a) for a in activations]
    fun.append(0)

    # Indices for weight vectorization following aenet's conventions
    iw = np.zeros(nlayers, dtype=int)
    iv = np.zeros(nlayers, dtype=int)
    wsize = 0
    nvalues = 0
    for il in range(0, nlayers - 1):
        wsize += (nnodes[il] + 1) * nnodes[il + 1]
        iw[il + 1] = wsize
        nvalues += nnodes[il] + 1
        iv[il + 1] = nvalues
    nvalues += nnodes[-1]

    # Flatten weights/biases into Fortran-order vector
    W_list: List[float] = []
    # Iterate over Linear layers in order; skip activations
    linears: List[nn.Linear] = [m for m in seq if isinstance(m, nn.Linear)]
    for lin in linears:
        # Match aenet-PyTorch output_nn.py:
        # weight: (nnodes2, nnodes1) == (out, in)
        # bias:   (nnodes2,)
        weight = lin.weight.detach().cpu().numpy()
        bias = lin.bias.detach().cpu().numpy().reshape(-1, 1)  # (out,1)
        # Concatenate [W | b] to shape (out, in+1) and flatten Fortran-order
        wb = np.concatenate([weight, bias], axis=1)
        W_list.extend(np.reshape(
            wb, ((wb.shape[0] * wb.shape[1]),), order="F").tolist())
    W = np.asarray(W_list, dtype=np.float64)

    # Write textual section identical to aenet-pytorch's formatting
    f.write("{:17d}\n".format(nlayers))
    f.write("{:17d}\n".format(nnodesmax))
    f.write("{:17d}\n".format(wsize))
    f.write("{:17d}\n".format(nvalues))
    fmt_i_nodes = "{:17d} " * nlayers + "\n"
    f.write(fmt_i_nodes.format(*nnodes))
    fmt_i_fun = "{:17d} " * (nlayers - 1) + "\n"
    f.write(fmt_i_fun.format(*fun))
    fmt_iw = "{:17d} " * nlayers + "\n"
    f.write(fmt_iw.format(*iw.tolist()))
    f.write(fmt_iw.format(*iv.tolist()))
    fmt_w = "{:24.17f} " * wsize + "\n"
    f.write(fmt_w.format(*W.tolist()))


def _write_setup_section(
    f: io.TextIOBase,
    trainer,
    species_index: int,
    species_name: str,
    stats: _DescriptorStats,
) -> None:
    """
    Write the descriptor setup section matching Fortran save_Setup_ASCII.

    For ChebyshevDescriptor the Fortran side expects:
      nsfparam = 4
      sfparam(1,1) = radial_Rc
      sfparam(2,1) = radial_N
      sfparam(3,1) = angular_Rc
      sfparam(4,1) = angular_N
    The remainder can be zeros for compatibility.
    """
    desc = trainer.descriptor
    species_all = _get_species_list(trainer)
    nenv = len(species_all)
    nsf = int(desc.get_n_features())
    nsfparam = 4

    # Prepare arrays
    sf = np.zeros((nsf,), dtype=int)
    sfparam = np.zeros((nsfparam, nsf), dtype=np.float64)
    sfenv = np.zeros((2, nsf), dtype=int)

    # Fill first column with Chebyshev params
    try:
        sfparam[0, 0] = float(desc.rad_cutoff)
        sfparam[1, 0] = float(desc.rad_order)
        sfparam[2, 0] = float(desc.ang_cutoff)
        sfparam[3, 0] = float(desc.ang_order)
    except Exception:
        # If descriptor lacks these attributes, keep zeros
        pass

    # neval: if stats came from structures, use total count proxy; else 0
    # We do not track exact evaluations; leave neval as 0.
    neval = 0

    # Write with same formatting as Fortran save_Setup_ASCII
    f.write("{}\n".format("PyTorch Chebyshev descriptor export"))
    f.write("{}\n".format(species_name))
    f.write("{:17d}\n".format(nenv))
    fmt_env = "{} " * nenv + "\n"
    f.write(fmt_env.format(*species_all))
    # rcmin/rcmax
    rcmin = float(getattr(desc, "min_cutoff", 0.0))
    rcmax = float(
        max(float(getattr(desc, "rad_cutoff", 0.0)),
            float(getattr(desc, "ang_cutoff", 0.0)))
    )
    f.write("{:24.17f}\n".format(rcmin))
    f.write("{:24.17f}\n".format(rcmax))
    f.write("{}\n".format("Chebyshev"))
    f.write("{:17d}\n".format(nsf))
    f.write("{:17d}\n".format(nsfparam))
    # sf
    fmt_sfi = "{:17d} " * nsf + "\n"
    f.write(fmt_sfi.format(*sf.tolist()))

    # Write sfparam in Fortran column-major stream with line wrapping
    def _write_col_major_float_matrix(mat: np.ndarray,
                                      nrows: int,
                                      ncols: int,
                                      per_line: int = 12) -> None:
        vals: list[float] = []
        for c in range(ncols):
            for r in range(nrows):
                vals.append(float(mat[r, c]))
        for i in range(0, len(vals), per_line):
            seg = vals[i:i + per_line]
            f.write("".join("{:24.17f} ".format(v)
                            for v in seg).rstrip() + "\n")

    _write_col_major_float_matrix(sfparam, nsfparam, nsf)

    # sfenv (2 x nsf) also in column-major stream
    def _write_col_major_int_matrix(mat: np.ndarray,
                                    nrows: int,
                                    ncols: int,
                                    per_line: int = 24) -> None:
        vals: list[int] = []
        for c in range(ncols):
            for r in range(nrows):
                vals.append(int(mat[r, c]))
        for i in range(0, len(vals), per_line):
            seg = vals[i:i + per_line]
            f.write("".join("{:17d} ".format(v) for v in seg).rstrip() + "\n")

    _write_col_major_int_matrix(sfenv, 2, nsf)
    # neval
    f.write("{:17d}\n".format(neval))
    # stats arrays
    fmt_sfv = "{:24.17f} " * nsf + "\n"
    f.write(fmt_sfv.format(*_as_numpy_1d(stats.min).tolist()))
    f.write(fmt_sfv.format(*_as_numpy_1d(stats.max).tolist()))
    f.write(fmt_sfv.format(*_as_numpy_1d(stats.avg).tolist()))
    f.write(fmt_sfv.format(*_as_numpy_1d(stats.cov).tolist()))


def _compute_trainset_metadata(
    trainer,
    structures: Optional[Sequence[Any]],
) -> Dict[str, Any]:
    """
    Compute training set metadata for the trainset info section.
    """
    species_all = _get_species_list(trainer)
    n_species = len(species_all)
    # Defaults
    meta: Dict[str, Any] = dict(
        filename="aenet-python",
        normalized=("yes" if trainer._normalizer
                    and trainer._normalizer.normalize_energy else "no"),
        E_scaling=float(getattr(trainer._normalizer, "E_scaling", 1.0)),
        E_shift=float(getattr(trainer._normalizer, "E_shift", 0.0)),
        N_species=n_species,
        sys_species=species_all,
        E_atomic=[0.0] * n_species,
        N_atom=0,
        N_struc=0,
        E_min=0.0,
        E_max=0.0,
        E_avg=0.0,
    )
    # E_atomic by species if available
    try:
        E_atomic_dict = getattr(trainer, "_E_atomic", None) or {}
        meta["E_atomic"] = [float(E_atomic_dict.get(s, 0.0))
                            for s in species_all]
    except Exception:
        pass

    if not structures or len(structures) == 0:
        return meta

    # Aggregate energies
    energy_target = getattr(trainer, "_energy_target", "cohesive")
    n_atoms_total = 0
    e_pa_list: List[float] = []
    for s in structures:
        n = int(s.n_atoms)
        e_total = float(s.energy)
        if energy_target == "cohesive" and meta["E_atomic"]:
            # Sum E_atomic per atom in structure
            e_atomic_sum = 0.0
            for sp in s.species:
                try:
                    idx = species_all.index(sp)
                    e_atomic_sum += float(meta["E_atomic"][idx])
                except Exception:
                    pass
            e_total = e_total - e_atomic_sum
        e_pa = e_total / max(1, n)
        e_pa_list.append(e_pa)
        n_atoms_total += n

    if e_pa_list:
        meta["N_atom"] = n_atoms_total
        meta["N_struc"] = len(structures)
        meta["E_min"] = float(np.min(e_pa_list))
        meta["E_max"] = float(np.max(e_pa_list))
        meta["E_avg"] = float(np.mean(e_pa_list))

    return meta


def _write_trainset_section(
    f: io.TextIOBase,
    meta: Dict[str, Any],
) -> None:
    """
    Write the trainset info section (ASCII) in the same order/format
    as external/aenet-pytorch's output_nn.py (save_trainset_info).
    """
    f.write("{}\n".format(meta.get("filename", "")))
    # Fortran expects a logical token (T/F), not "yes"/"no"
    normalized_val = meta.get("normalized", True)
    if isinstance(normalized_val, str):
        normalized_val_l = normalized_val.strip().lower()
        normalized_flag = ("T" if normalized_val_l
                           in ("yes", "true", "t", "1") else "F")
    else:
        normalized_flag = "T" if bool(normalized_val) else "F"
    f.write("{}\n".format(normalized_flag))
    f.write("{:24.17f}\n".format(float(meta.get("E_scaling", 1.0))))
    f.write("{:24.17f}\n".format(float(meta.get("E_shift", 0.0))))
    N_species = int(meta.get("N_species", 0))
    f.write("{:}\n".format(N_species))
    # species names
    fmt_sp = "{} " * N_species + "\n"
    f.write(fmt_sp.format(*list(meta.get("sys_species", []))))
    # atomic reference energies per species
    e_at = [float(x) for x in list(meta.get("E_atomic", []))]
    fmt_ea = "{:24.17f} " * len(e_at) + "\n"
    f.write(fmt_ea.format(*e_at))
    f.write("{:}\n".format(int(meta.get("N_atom", 0))))
    f.write("{:}\n".format(int(meta.get("N_struc", 0))))
    f.write("{:24.17f} {:24.17f} {:24.17f}\n".format(
        float(meta.get("E_min", 0.0)),
        float(meta.get("E_max", 0.0)),
        float(meta.get("E_avg", 0.0)),
    ))


def export_to_ascii_impl(
    trainer,
    output_dir: PathLike,
    prefix: str = "potential",
    descriptor_stats: Optional[Dict[str, np.ndarray]] = None,
    structures: Optional[Sequence[Any]] = None,
    compute_stats: bool = True,
) -> List[Path]:
    """
    Export trained model to per-species .nn.ascii files.

    Parameters
    ----------
    trainer : TorchANNPotential
        Trained model wrapper.
    output_dir : str | Path
        Destination directory for output files.
    prefix : str
        Filename prefix: '{prefix}.{SPECIES}.nn.ascii'.
    descriptor_stats : dict, optional
        Pre-computed stats with keys 'min','max','avg','cov' (1D arrays).
    structures : sequence, optional
        Dataset structures for computing descriptor statistics and trainset
        metadata. Strongly recommended for exact min/max.
    compute_stats : bool
        If True and 'descriptor_stats' not provided, compute stats from
        'structures'. If False, try to derive from trainer normalizer.

    Returns
    -------
    List[Path]
        Paths to created files (one per species).
    """
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)

    species_all = _get_species_list(trainer)
    input_size = _get_input_size(trainer)

    # Resolve descriptor statistics
    stats_obj: Optional[_DescriptorStats] = None
    if descriptor_stats is not None:
        stats_obj = _DescriptorStats(
            min=_as_numpy_1d(descriptor_stats.get("min")),
            max=_as_numpy_1d(descriptor_stats.get("max")),
            avg=_as_numpy_1d(descriptor_stats.get("avg")),
            cov=_as_numpy_1d(descriptor_stats.get("cov")),
        )
    elif compute_stats and structures is not None:
        stats_obj = _gather_descriptor_stats_from_structures(
            trainer, structures)
    else:
        stats_obj = _derive_descriptor_stats_from_normalizer(trainer)

    if stats_obj is None:
        raise RuntimeError(
            "Descriptor statistics unavailable. Provide 'structures' with "
            "compute_stats=True, or pass 'descriptor_stats' explicitly."
        )

    # Trainset metadata
    meta = _compute_trainset_metadata(trainer, structures)

    # Write one file per species
    out_paths: List[Path] = []
    for isp, sp in enumerate(species_all):
        out_path = out_dir / f"{prefix}.{sp}.nn.ascii"
        with open(out_path, "w", encoding="utf-8") as f:
            _write_network_section(f, trainer, isp, sp, input_size)
            _write_setup_section(f, trainer, isp, sp, stats_obj)
            _write_trainset_section(f, meta)
        out_paths.append(out_path)

    return out_paths
