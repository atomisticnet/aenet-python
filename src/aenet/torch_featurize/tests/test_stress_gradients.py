"""
Stress tests for gradient stability under small cells and high neighbor
counts.

Focus:
- Very small periodic cells leading to many periodic images within cutoff
- Dense FCC-like cells
- Multi-species mode engaged (typespin paths)
- Tight angular cutoffs

These tests primarily check numerical stability: no NaN/Inf and
reasonable magnitudes. They also spot-check a finite-difference gradient
on a representative coordinate.
"""
import torch
import pytest

from aenet.torch_featurize.featurize import ChebyshevDescriptor


def _finite_diff_grad(
    descriptor,
    positions,
    species,
    cell,
    pbc,
    atom_idx,
    coord_idx,
    feature_idx,
    eps=1e-5,
):
    """Central difference finite-difference for a single feature component."""
    pos_f = positions.clone()
    pos_f[atom_idx, coord_idx] += eps
    feat_f = descriptor.forward_from_positions(pos_f, species, cell, pbc)

    pos_b = positions.clone()
    pos_b[atom_idx, coord_idx] -= eps
    feat_b = descriptor.forward_from_positions(pos_b, species, cell, pbc)

    return (feat_f[atom_idx, feature_idx] - feat_b[atom_idx, feature_idx]) / (
        2.0 * eps
    )


class TestStressGradients:
    @pytest.mark.parametrize(
        "alat, rad_cut, ang_cut",
        [
            (1.5, 4.0, 3.0),  # many images within cutoff
            (1.8, 4.0, 3.0),  # slightly larger, still dense
        ],
    )
    def test_tiny_cubic_many_images(self, alat, rad_cut, ang_cut):
        """
        Single atom in a very small periodic cubic cell -> many periodic
        images. Ensures no NaNs/Infs and gradients remain bounded.
        """
        device = "cpu"
        dtype = torch.float64
        cell = torch.eye(3, dtype=dtype) * alat
        pbc = torch.tensor([True, True, True])
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype)
        species = ["H"]

        descriptor = ChebyshevDescriptor(
            species=["H"],
            rad_order=5,
            rad_cutoff=rad_cut,
            ang_order=3,
            ang_cutoff=ang_cut,
            device=device,
            dtype=dtype,
        )

        features, grads = descriptor.compute_feature_gradients(
            positions, species, cell, pbc
        )

        # Basic sanity
        assert torch.isfinite(features).all(), (
            "Features contain non-finite values"
        )
        assert torch.isfinite(grads).all(), (
            "Gradients contain non-finite values"
        )

        # Bound magnitudes (very generous upper bound to avoid false positives)
        max_abs = grads.abs().max().item() if grads.numel() else 0.0
        assert max_abs < 1e10, f"Gradient magnitude exploded: {max_abs:g}"

        # Spot-check a finite difference on first feature if available
        if features.shape[1] > 0:
            fd = _finite_diff_grad(
                descriptor,
                positions,
                species,
                cell,
                pbc,
                atom_idx=0,
                coord_idx=0,
                feature_idx=0,
                eps=1e-6,
            )
            an = grads[0, 0, 0, 0]
            abs_diff = (an - fd).abs()
            rel_diff = abs_diff / (fd.abs() + 1e-12)
            msg = (
                "FD mismatch (tiny cubic): "
                f"analytical={an:.6e}, fd={fd:.6e}, "
                f"abs_diff={abs_diff:.2e}, rel_diff={rel_diff:.2%}"
            )
            assert abs_diff < 1e-3 or rel_diff < 0.05, msg

    def test_dense_fcc_small_a(self):
        """
        Dense FCC-like lattice with small 'a' -> high neighbor count.
        Uses the rotated FCC basis from existing tests but smaller scale.
        """
        dtype = torch.float64
        device = "cpu"

        # Base FCC lattice vectors (alat = 2.0) then rotate (from existing
        # tests)
        alat = 2.0
        avec = torch.tensor(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            dtype=dtype,
        ) * alat

        ax = torch.sqrt(torch.tensor(0.3, dtype=dtype))
        R1 = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, ax, ax],
                [0.0, -ax, ax],
            ],
            dtype=dtype,
        )
        avec = R1 @ avec

        ax = torch.sqrt(torch.tensor(0.7, dtype=dtype))
        R2 = torch.tensor(
            [
                [ax, 0.0, ax],
                [0.0, 1.0, 0.0],
                [-ax, 0.0, ax],
            ],
            dtype=dtype,
        )
        cell = R2 @ avec

        pbc = torch.tensor([True, True, True])
        positions = torch.tensor([[0.2, -0.1, 0.15]], dtype=dtype)
        species = ["H"]

        descriptor = ChebyshevDescriptor(
            species=["H"],
            rad_order=6,
            rad_cutoff=4.0,
            ang_order=4,
            ang_cutoff=3.0,
            device=device,
            dtype=dtype,
        )

        features, grads = descriptor.compute_feature_gradients(
            positions, species, cell, pbc
        )
        assert torch.isfinite(features).all()
        assert torch.isfinite(grads).all()
        max_abs = grads.abs().max().item() if grads.numel() else 0.0
        assert max_abs < 1e10, f"Gradient magnitude exploded: {max_abs:g}"

        # Spot-check finite difference for a mid feature if available
        F = features.shape[1]
        if F > 2:
            feature_idx = min(2, F - 1)
            fd = _finite_diff_grad(
                descriptor,
                positions,
                species,
                cell,
                pbc,
                atom_idx=0,
                coord_idx=1,
                feature_idx=feature_idx,
                eps=1e-6,
            )
            an = grads[0, feature_idx, 0, 1]
            abs_diff = (an - fd).abs()
            rel_diff = abs_diff / (fd.abs() + 1e-12)
            msg = (
                "FD mismatch (dense FCC): "
                f"analytical={an:.6e}, fd={fd:.6e}, "
                f"abs_diff={abs_diff:.2e}, rel_diff={rel_diff:.2%}"
            )
            assert abs_diff < 2e-3 or rel_diff < 0.08, msg

    def test_multi_species_mode_tight_cutoff(self):
        """
        Engage multi-species code paths and test tight angular cutoff.
        Single atom of species 'A' in a tiny cubic PBC cell is sufficient to
        trigger typespin-weighted accumulation via periodic images.
        """
        dtype = torch.float64
        device = "cpu"

        # Tiny cell; nearest image at 1.6 Å will be inside rad/ang cutoffs.
        cell = torch.eye(3, dtype=dtype) * 1.6
        pbc = torch.tensor([True, True, True])
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype)
        species = ["A"]  # Only A present; multi-species enabled by descriptor

        descriptor = ChebyshevDescriptor(
            species=["A", "B"],  # enable multi-species/typespin
            rad_order=5,
            rad_cutoff=2.5,
            ang_order=3,
            ang_cutoff=2.0,  # relatively tight
            device=device,
            dtype=dtype,
        )

        features, grads = descriptor.compute_feature_gradients(
            positions, species, cell, pbc
        )
        assert torch.isfinite(features).all()
        assert torch.isfinite(grads).all()
        max_abs = grads.abs().max().item() if grads.numel() else 0.0
        assert max_abs < 1e10, f"Gradient magnitude exploded: {max_abs:g}"

        # Spot-check one FD entry
        if features.shape[1] > 0:
            fd = _finite_diff_grad(
                descriptor,
                positions,
                species,
                cell,
                pbc,
                atom_idx=0,
                coord_idx=2,
                feature_idx=0,
                eps=1e-6,
            )
            an = grads[0, 0, 0, 2]
            abs_diff = (an - fd).abs()
            rel_diff = abs_diff / (fd.abs() + 1e-12)
            msg = (
                "FD mismatch (multi tight): "
                f"analytical={an:.6e}, fd={fd:.6e}, "
                f"abs_diff={abs_diff:.2e}, rel_diff={rel_diff:.2%}"
            )
            assert abs_diff < 2e-3 or rel_diff < 0.08, msg
