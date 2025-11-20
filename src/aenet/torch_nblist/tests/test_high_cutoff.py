"""
High-cutoff coverage tests for TorchNeighborList (ghost backend).

- Validates analytical neighbor counts for a one-atom cubic cell at
  several cutoffs, including ranges that require replication beyond
  the immediate ±1 image layers (i.e., beyond "27 cells").
- Validates PBC single-cell neighbor distances against an explicit
  supercell central-atom neighborhood under a high cutoff.
"""

import math
import torch
import pytest

from aenet.torch_nblist import TorchNeighborList


def _num_neighbors_cubic_single_atom(a: float, cutoff: float) -> int:
    """
    Analytical neighbor count for a single atom in a cubic lattice
    with lattice constant a and cutoff in Angstroms.

    Counts integer lattice vectors (h,k,l) != (0,0,0) whose length
    |(h,k,l)| * a <= cutoff.

    This is used here to produce expected counts for small shells.
    """
    # We only need a small range around the origin sufficient for given cutoff
    max_h = int(math.ceil(cutoff / a)) + 1
    cnt = 0
    for h in range(-max_h, max_h + 1):
        for k in range(-max_h, max_h + 1):
            for lz in range(-max_h, max_h + 1):
                if h == 0 and k == 0 and lz == 0:
                    continue
                d = a * math.sqrt(h * h + k * k + lz * lz)
                if d <= cutoff + 1e-12:
                    cnt += 1
    return cnt


class TestHighCutoffCubicCounts:
    """
    Check analytical neighbor counts for cubic single-atom system under PBC.
    """

    @pytest.mark.parametrize(
        "a, cutoff, expected",
        [
            # a = 1.0
            # r = 1.1: only axis neighbors at distance = 1.0 -> 6
            (1.0, 1.1, 6),
            # r = sqrt(2) + 0.1: include axis (6) + face diagonals (12) -> 18
            (1.0, math.sqrt(2.0) + 0.1, 18),
            # r = 2.1: include axis (6), face (12), body (8),
            #           and (2,0,0) (6) -> 32
            (1.0, 2.1, 32),
        ],
    )
    def test_cubic_single_atom_counts(self, a, cutoff, expected):
        device = "cpu"
        dtype = torch.float64

        # Single atom at origin, fractional coordinates
        positions_frac = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype)
        cell = torch.eye(3, dtype=dtype) * a
        pbc = torch.tensor([True, True, True], dtype=torch.bool)

        nbl = TorchNeighborList(cutoff=cutoff, device=device, dtype=dtype)
        result = nbl.get_neighbors(
            positions_frac, cell=cell, pbc=pbc, fractional=True
        )

        num_neighbors = int(result["num_neighbors"][0].item())

        # Sanity check vs analytical count for these small shells
        assert num_neighbors == expected, (
            f"Expected {expected}, got {num_neighbors} "
            f"(a={a}, cutoff={cutoff})"
        )

    def test_cubic_high_cutoff_pbc_vs_supercell(self):
        """
        Compare PBC single-cell vs explicit 5x5x5 supercell central atom
        for a=1.0 and cutoff=2.1 (requires replication beyond ±1).
        """
        device = "cpu"
        dtype = torch.float64

        a = 1.0
        cutoff = 2.1

        cell = torch.eye(3, dtype=dtype) * a
        pbc = torch.tensor([True, True, True], dtype=torch.bool)

        nbl = TorchNeighborList(cutoff=cutoff, device=device, dtype=dtype)

        # PBC single-cell
        pos_pbc = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype)
        data_pbc = nbl.get_neighbors(
            pos_pbc, cell=cell, pbc=pbc, fractional=False
        )
        i_pbc = data_pbc["edge_index"][0]
        j_pbc = data_pbc["edge_index"][1]
        off_pbc = data_pbc["offsets"].to(dtype)

        r_pbc = (pos_pbc[j_pbc] + off_pbc @ cell) - pos_pbc[i_pbc]
        d_pbc = torch.norm(r_pbc, dim=-1)

        # Explicit 5x5x5 supercell.
        # Center index = 62 in row-major.
        # For 3x3x3 (-1..+1) the center would be 13;
        # for 5x5x5 (-2..+2) it's 62.
        shifts = []
        for h in range(-2, 3):
            for k in range(-2, 3):
                for lz in range(-2, 3):
                    shifts.append((h, k, lz))
        shifts = torch.tensor(shifts, dtype=dtype)
        pos_sup = shifts @ cell  # (125,3)
        # supercell size is implicit in the generated positions (5x5x5 grid)

        # For explicit supercell comparison, do NOT apply PBC here.
        # We want neighbors within the constructed supercell only.
        data_sup = nbl.get_neighbors(
            pos_sup, cell=None, pbc=None, fractional=False
        )
        i_sup = data_sup["edge_index"][0]
        j_sup = data_sup["edge_index"][1]

        r_sup = pos_sup[j_sup] - pos_sup[i_sup]
        d_sup = torch.norm(r_sup, dim=-1)

        # Central atom index (2,2,2) in -2..+2 grid.
        # Linear index:
        # idx = (2)*25 + (2)*5 + (2) = 62
        central_idx = 2 * 25 + 2 * 5 + 2
        mask_center = (i_sup == central_idx)
        d_center = d_sup[mask_center]

        # Compare histograms of distances (rounded to 6 dp) within cutoff
        def rounded_hist(d):
            d = torch.round(d * 1_000_000) / 1_000_000
            uniq, counts = torch.unique(d, return_counts=True)
            # Convert to dict for easy compare
            return {float(u.item()): int(c.item())
                    for u, c in zip(uniq, counts)}

        hist_pbc = rounded_hist(d_pbc)
        hist_center = rounded_hist(d_center)

        # They should match exactly for this simple cubic case
        assert hist_pbc == hist_center, (
            "Distance histograms mismatch:\n"
            f"PBC: {hist_pbc}\nCenter: {hist_center}"
        )

        # And total neighbor count should match analytical
        expected = _num_neighbors_cubic_single_atom(a, cutoff)
        assert len(d_pbc) == expected, (
            f"Expected {expected} PBC neighbors, got {len(d_pbc)}"
        )
        assert len(d_center) == expected, (
            f"Expected {expected} center neighbors, got {len(d_center)}"
        )
