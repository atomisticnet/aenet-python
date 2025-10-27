#!/usr/bin/env python
"""
Profile feature and gradient performance for ChebyshevDescriptor.

This script measures the computational performance of the Chebyshev
descriptor's feature and gradient computations on periodic systems of
varying sizes. It helps identify performance bottlenecks and guide
optimization efforts.

Usage Examples
--------------
Basic profiling with default settings (2×2×2, 4×4×4, 6×6×6 supercells):
  profile_gradients.py

Profile specific sizes with more repeats for statistical reliability:
  profile_gradients.py --sizes 2 4 6 8 --repeats 5

Test single-species mode (disables typespin-weighted features):
  profile_gradients.py --sizes 2 4 --single-species

Adjust descriptor parameters:
  profile_gradients.py --rad-order 10 --ang-order 5

What It Does
------------
1. Generates simple-cubic periodic supercells (N×N×N atoms)
2. Applies small random jitter to avoid perfect symmetry artifacts
3. Counts pairs (for radial features) and triplets (for angular features)
4. Times feature computation over multiple runs
5. Times gradient computation over multiple runs
6. Reports mean/min/max timing statistics

Output Columns
--------------
- N: Supercell edge size (N×N×N atoms)
- atoms: Total number of atoms in the cell
- pairs: Number of (i,j) neighbor pairs within radial cutoff
- triplets: Number of (i,j,k) neighbor triplets within angular cutoff
- feat_mean: Mean feature computation time (seconds)
- grad_mean: Mean gradient computation time (seconds)
- feat[min,max]: Range of feature times across repeats
- grad[min,max]: Range of gradient times across repeats

Performance Notes
-----------------
- Gradient computation is typically 1.5-3× slower than features alone
- Scales roughly as O(N_atoms × N_neighbors²) for angular features
- Current implementation loops over n_ang features; batched scatter
  could remove this loop if profiling shows it's a bottleneck
"""
from __future__ import annotations

import argparse
import time
from typing import List, Tuple

import torch

from aenet.torch_featurize.featurize import ChebyshevDescriptor

__author__ = "The aenet developers"
__email__ = "aenet@atomistic.net"
__date__ = "2025-10-27"
__version__ = "0.1"


def build_sc_cell(n: int,
                  alat: float,
                  dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build simple cubic lattice with n^3 atoms; positions on lattice points.

    Returns:
        positions: (N, 3)
        cell: (3, 3)
    """
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
    """Apply small random jitter to avoid perfect symmetry artifacts."""
    if scale <= 0:
        return positions
    noise = (torch.rand_like(positions) - 0.5) * 2.0 * scale
    return positions + noise


def count_pairs_triplets(positions: torch.Tensor,
                         cell: torch.Tensor,
                         pbc: torch.Tensor,
                         descriptor: ChebyshevDescriptor) -> Tuple[int, int]:
    """
    Count number of (i,j) pairs and (i,j,k) triplets used by
    radial/angular features. Using the internal neighbor list with
    the max cutoff.
    """
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

    # Triplets per center: we need per-center neighbors within angular cutoff
    i_idx = edge_index[0]
    j_idx = edge_index[1]

    # Compute displacement vectors with offsets
    if cell is not None and offsets is not None:
        r_ij = (positions[j_idx] + offsets.to(descriptor.dtype) @ cell
                ) - positions[i_idx]
    else:
        r_ij = positions[j_idx] - positions[i_idx]
    d_ij = torch.norm(r_ij, dim=-1)

    ang_mask = (d_ij <= descriptor.ang_cutoff) & (d_ij > descriptor.min_cutoff)
    i_idx_ang = i_idx[ang_mask]

    # Count combinations per center: C(m,2)
    n_triplets = 0
    if len(i_idx_ang) > 0:
        # bincount over centers
        max_i = int(i_idx_ang.max().item()) + 1
        counts = torch.bincount(
            i_idx_ang, minlength=max(positions.shape[0], max_i))
        # combinations per center
        n_triplets = int(torch.sum(counts * (counts - 1) // 2).item())

    return n_pairs, n_triplets


def profile_once(n: int, args: argparse.Namespace) -> dict:
    dtype = torch.float64
    device = args.device

    # Lattice parameter chosen to keep reasonable neighbor counts at cutoffs
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
        "pairs": n_pairs,
        "triplets": n_triplets,
        "feat_mean": sum(feat_times) / len(feat_times),
        "feat_min": min(feat_times),
        "feat_max": max(feat_times),
        "grad_mean": sum(grad_times) / len(grad_times),
        "grad_min": min(grad_times),
        "grad_max": max(grad_times),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(__doc__+"\n{} {}".format(__date__, __author__)),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 6],
                        help="List of supercell edge sizes N for NxNxN atoms.")
    parser.add_argument("--alat", type=float, default=2.5,
                        help="Lattice parameter (Angstrom).")
    parser.add_argument("--rad-cutoff", type=float,
                        default=4.0, dest="rad_cutoff",
                        help="Radial cutoff radius.")
    parser.add_argument("--ang-cutoff", type=float,
                        default=3.0, dest="ang_cutoff",
                        help="Angular cutoff radius.")
    parser.add_argument("--rad-order", type=int, default=5, dest="rad_order")
    parser.add_argument("--ang-order", type=int, default=3, dest="ang_order")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Timing repeats per size.")
    parser.add_argument("--jitter", type=float, default=1e-3,
                        help="Random displacement scale.")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu"], help="Device.")
    parser.add_argument("--single-species", action="store_true",
                        help="Use single-species mode; "
                             + "default is multi-species.")
    args = parser.parse_args()

    torch.set_num_threads(max(torch.get_num_threads(), 1))

    print("Config:")
    print(f"  sizes         : {args.sizes}")
    print(f"  alat          : {args.alat}")
    print(f"  rad_cutoff    : {args.rad_cutoff}")
    print(f"  ang_cutoff    : {args.ang_cutoff}")
    print(f"  rad_order     : {args.rad_order}")
    print(f"  ang_order     : {args.ang_order}")
    print(f"  repeats       : {args.repeats}")
    print(f"  jitter        : {args.jitter}")
    print(f"  device        : {args.device}")
    print(f"  single_species: {args.single_species}")
    print("")

    header = ("N  atoms    pairs      triplets    feat_mean(s)  grad_mean(s)  "
              "feat[min,max]  grad[min,max]")
    print(header)
    print("-" * len(header))

    for n in args.sizes:
        res = profile_once(n, args)
        print(
            f"{res['n']:>1}  {res['atoms']:>5}  "
            f"{res['pairs']:>9}  {res['triplets']:>11}  "
            f"{res['feat_mean']:>12.6f}  {res['grad_mean']:>12.6f}  "
            f"[{res['feat_min']:.6f},{res['feat_max']:.6f}]  "
            f"[{res['grad_min']:.6f},{res['grad_max']:.6f}]"
        )


if __name__ == "__main__":
    main()
