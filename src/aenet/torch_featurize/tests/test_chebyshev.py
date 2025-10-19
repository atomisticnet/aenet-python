"""
Unit tests for Chebyshev polynomial evaluation and symmetry functions.

Tests numerical accuracy, derivatives, and comparison with recurrence relation.
"""

import pytest
import torch
from aenet.torch_featurize.chebyshev import (
    ChebyshevPolynomials,
    RadialBasis,
    AngularBasis
)


class TestChebyshevPolynomials:
    """Test Chebyshev polynomial evaluation."""

    def test_rescaling_boundaries(self):
        """Test distance rescaling at boundary values."""
        cheb = ChebyshevPolynomials(max_order=5, r_min=0.5, r_max=4.0)

        # Test boundary values
        r_min = torch.tensor([0.5], dtype=torch.float64)
        r_max = torch.tensor([4.0], dtype=torch.float64)
        r_mid = torch.tensor([2.25], dtype=torch.float64)

        x_min = cheb.rescale_distances(r_min)
        x_max = cheb.rescale_distances(r_max)
        x_mid = cheb.rescale_distances(r_mid)

        assert torch.allclose(x_min, torch.tensor([-1.0], dtype=torch.float64))
        assert torch.allclose(x_max, torch.tensor([1.0], dtype=torch.float64))
        assert torch.allclose(x_mid, torch.tensor([0.0], dtype=torch.float64))

    def test_rescaling_clamping(self):
        """Test that rescaling clamps values outside [r_min, r_max]."""
        cheb = ChebyshevPolynomials(max_order=5, r_min=0.5, r_max=4.0)

        # Values outside range should be clamped
        r_below = torch.tensor([0.1], dtype=torch.float64)
        r_above = torch.tensor([5.0], dtype=torch.float64)

        x_below = cheb.rescale_distances(r_below)
        x_above = cheb.rescale_distances(r_above)

        assert torch.allclose(
            x_below, torch.tensor([-1.0], dtype=torch.float64))
        assert torch.allclose(
            x_above, torch.tensor([1.0], dtype=torch.float64))

    def test_known_chebyshev_values(self):
        """Test against known Chebyshev polynomial values."""
        cheb = ChebyshevPolynomials(max_order=3, r_min=-1.0, r_max=1.0)

        # Known values: T_0(0.5)=1, T_1(0.5)=0.5, T_2(0.5)=-0.5, T_3(0.5)=-1.0
        x = torch.tensor([0.5], dtype=torch.float64)
        T = cheb(x)

        expected = torch.tensor([[1.0, 0.5, -0.5, -1.0]], dtype=torch.float64)
        assert torch.allclose(T, expected, atol=1e-10)

    def test_known_values_at_zero(self):
        """Test Chebyshev values at x=0."""
        cheb = ChebyshevPolynomials(max_order=4, r_min=-1.0, r_max=1.0)

        x = torch.tensor([0.0], dtype=torch.float64)
        T = cheb(x)

        # T_0(0)=1, T_1(0)=0, T_2(0)=-1, T_3(0)=0, T_4(0)=1
        expected = torch.tensor(
            [[1.0, 0.0, -1.0, 0.0, 1.0]], dtype=torch.float64)
        assert torch.allclose(T, expected, atol=1e-10)

    def test_known_values_at_extrema(self):
        """Test Chebyshev values at x=-1 and x=1."""
        cheb = ChebyshevPolynomials(max_order=5, r_min=-1.0, r_max=1.0)

        x_neg = torch.tensor([-1.0], dtype=torch.float64)
        x_pos = torch.tensor([1.0], dtype=torch.float64)

        T_neg = cheb(x_neg)
        T_pos = cheb(x_pos)

        # At x=1: T_n(1) = 1 for all n
        expected_pos = torch.ones(1, 6, dtype=torch.float64)
        assert torch.allclose(T_pos, expected_pos, atol=1e-10)

        # At x=-1: T_n(-1) = (-1)^n
        expected_neg = torch.tensor(
            [[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]], dtype=torch.float64)
        assert torch.allclose(T_neg, expected_neg, atol=1e-10)

    def test_recurrence_equivalence(self):
        """Verify cosine form matches recurrence relation."""
        max_order = 10
        cheb = ChebyshevPolynomials(max_order=max_order, r_min=-1.0, r_max=1.0)

        # Test points (avoid exact ±1 for numerical stability)
        x_test = torch.linspace(-0.99, 0.99, 20, dtype=torch.float64)

        # Cosine form
        T_cos = cheb(x_test)

        # Recurrence form
        T_rec = torch.zeros(len(x_test), max_order + 1, dtype=torch.float64)
        T_rec[:, 0] = 1.0
        if max_order >= 1:
            T_rec[:, 1] = x_test
        for n in range(2, max_order + 1):
            T_rec[:, n] = 2.0 * x_test * T_rec[:, n - 1] - T_rec[:, n - 2]

        assert torch.allclose(T_cos, T_rec, atol=1e-12)

    def test_batch_processing(self):
        """Test that batched inputs work correctly."""
        cheb = ChebyshevPolynomials(max_order=5, r_min=0.5, r_max=4.0)

        # Batch of distances
        r_batch = torch.tensor([1.0, 2.0, 3.0, 3.5], dtype=torch.float64)
        T_batch = cheb(r_batch)

        assert T_batch.shape == (4, 6)  # 4 distances, 6 orders (0-5)

        # Verify each element independently
        for i, r in enumerate(r_batch):
            T_single = cheb(r.unsqueeze(0))
            assert torch.allclose(T_batch[i], T_single[0], atol=1e-12)

    def test_derivatives_finite_difference(self):
        """Test derivatives against finite differences."""
        cheb = ChebyshevPolynomials(max_order=5, r_min=0.5, r_max=4.0)

        r = torch.tensor([1.5, 2.0, 3.0], dtype=torch.float64)
        eps = 1e-6

        # Analytical derivatives
        T, dT_dr = cheb.evaluate_with_derivatives(r)

        # Finite difference derivatives
        r_plus = r + eps
        r_minus = r - eps
        T_plus = cheb(r_plus)
        T_minus = cheb(r_minus)
        dT_dr_fd = (T_plus - T_minus) / (2 * eps)

        # Check agreement (finite differences less accurate than analytical)
        assert torch.allclose(dT_dr, dT_dr_fd, rtol=1e-5, atol=1e-5)

    def test_derivatives_autodiff(self):
        """Test derivatives using PyTorch autodiff."""
        cheb = ChebyshevPolynomials(max_order=5, r_min=0.5, r_max=4.0)

        r = torch.tensor(
            [1.5, 2.0, 3.0], dtype=torch.float64, requires_grad=True)

        # Compute with autodiff
        T = cheb(r)
        T_sum = T.sum()
        T_sum.backward()
        grad_autodiff = r.grad.clone()

        # Compute with implemented derivatives
        r_no_grad = r.detach().clone()
        T, dT_dr = cheb.evaluate_with_derivatives(r_no_grad)
        grad_implemented = dT_dr.sum(dim=-1)

        assert torch.allclose(grad_autodiff, grad_implemented, atol=1e-10)

    def test_derivative_at_boundary(self):
        """Test derivative at order 0 is zero."""
        cheb = ChebyshevPolynomials(max_order=5, r_min=0.5, r_max=4.0)

        r = torch.tensor([1.5, 2.0, 3.0], dtype=torch.float64)
        T, dT_dr = cheb.evaluate_with_derivatives(r)

        # T_0(x) = 1, so dT_0/dr = 0
        assert torch.allclose(dT_dr[:, 0], torch.zeros(3, dtype=torch.float64))

    def test_cutoff_function_values(self):
        """Test cutoff function values."""
        cheb = ChebyshevPolynomials(max_order=5, r_min=0.5, r_max=4.0)
        Rc = 3.0

        # Test at various points
        r = torch.tensor([0.0, 1.5, 3.0, 4.0], dtype=torch.float64)
        fc = cheb.cutoff_function(r, Rc)

        # At r=0: fc = 0.5 * (cos(0) + 1) = 1.0
        assert torch.allclose(fc[0], torch.tensor(1.0, dtype=torch.float64))

        # At r=Rc/2: fc = 0.5 * (cos(π/2) + 1) = 0.5
        assert torch.allclose(
            fc[1], torch.tensor(0.5, dtype=torch.float64), atol=1e-10)

        # At r=Rc: fc = 0.5 * (cos(π) + 1) = 0.0
        assert torch.allclose(
            fc[2], torch.tensor(0.0, dtype=torch.float64), atol=1e-10)

        # At r>Rc: fc = 0.0
        assert torch.allclose(fc[3], torch.tensor(0.0, dtype=torch.float64))

    def test_cutoff_derivative_values(self):
        """Test cutoff derivative values."""
        cheb = ChebyshevPolynomials(max_order=5, r_min=0.5, r_max=4.0)
        Rc = 3.0

        r = torch.tensor([0.0, 3.0, 4.0], dtype=torch.float64)
        dfc = cheb.cutoff_derivative(r, Rc)

        # At r=0: dfc/dr = -0.5 * π/Rc * sin(0) = 0
        assert torch.allclose(dfc[0], torch.tensor(0.0, dtype=torch.float64))

        # At r=Rc: dfc/dr = -0.5 * π/Rc * sin(π) = 0
        assert torch.allclose(
            dfc[1], torch.tensor(0.0, dtype=torch.float64), atol=1e-10)

        # At r>Rc: dfc/dr = 0
        assert torch.allclose(dfc[2], torch.tensor(0.0, dtype=torch.float64))

    def test_cutoff_continuity(self):
        """Test continuity of cutoff function at Rc."""
        cheb = ChebyshevPolynomials(max_order=5, r_min=0.5, r_max=4.0)
        Rc = 3.0
        eps = 1e-8

        # Points just before and after Rc
        r_before = torch.tensor([Rc - eps], dtype=torch.float64)
        r_after = torch.tensor([Rc + eps], dtype=torch.float64)

        fc_before = cheb.cutoff_function(r_before, Rc)
        fc_after = cheb.cutoff_function(r_after, Rc)

        # Should be very close to 0
        assert torch.allclose(
            fc_before, torch.zeros_like(fc_before), atol=1e-6)
        assert torch.allclose(fc_after, torch.zeros_like(fc_after), atol=1e-10)

    @pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_compatibility(self):
        """Test that module works on GPU."""
        cheb = ChebyshevPolynomials(max_order=5, r_min=0.5, r_max=4.0)
        cheb = cheb.cuda()

        r = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device='cuda')
        T = cheb(r)

        assert T.device.type == 'cuda'
        assert T.shape == (3, 6)


class TestRadialBasis:
    """Test radial basis functions."""

    def test_basic_evaluation(self):
        """Test basic radial feature evaluation."""
        rad_sf = RadialBasis(rad_order=10, rad_cutoff=4.0)

        distances = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        G_rad = rad_sf(distances)

        assert G_rad.shape == (3, 11)  # 3 distances, 11 orders (0-10)

    def test_cutoff_application(self):
        """Test that cutoff is properly applied."""
        rad_sf = RadialBasis(rad_order=5, rad_cutoff=3.0)

        # Distance beyond cutoff
        r_beyond = torch.tensor([4.0], dtype=torch.float64)
        G_rad = rad_sf(r_beyond)

        # Should be all zeros
        assert torch.allclose(G_rad, torch.zeros_like(G_rad))

    def test_product_rule_derivatives(self):
        """Test that derivatives follow product rule."""
        rad_sf = RadialBasis(rad_order=5, rad_cutoff=4.0)

        distances = torch.tensor([1.5, 2.0, 3.0], dtype=torch.float64)
        G_rad, dG_rad = rad_sf.forward_with_derivatives(distances)

        # Verify against finite differences
        eps = 1e-6
        distances_plus = distances + eps
        distances_minus = distances - eps
        G_plus = rad_sf(distances_plus)
        G_minus = rad_sf(distances_minus)
        dG_fd = (G_plus - G_minus) / (2 * eps)

        assert torch.allclose(dG_rad, dG_fd, rtol=1e-5, atol=1e-5)

    def test_zero_at_cutoff(self):
        """Test that features go to zero at cutoff."""
        rad_sf = RadialBasis(rad_order=5, rad_cutoff=3.0)

        r_at_cutoff = torch.tensor([3.0], dtype=torch.float64)
        G_rad = rad_sf(r_at_cutoff)

        # Should be approximately zero (cutoff function = 0)
        assert torch.allclose(G_rad, torch.zeros_like(G_rad), atol=1e-10)

    def test_min_cutoff(self):
        """Test minimum cutoff handling."""
        rad_sf = RadialBasis(
            rad_order=5, rad_cutoff=4.0, min_cutoff=0.8
        )

        # Distance below min_cutoff should still be handled
        r_below = torch.tensor([0.5], dtype=torch.float64)
        G_rad = rad_sf(r_below)

        # Should not crash and should have some values
        assert G_rad.shape == (1, 6)


class TestAngularBasis:
    """Test angular basis functions."""

    def test_basic_evaluation(self):
        """Test basic angular feature evaluation."""
        ang_sf = AngularBasis(ang_order=3, ang_cutoff=2.0)

        # Example triplet: distances and angle
        r_ij = torch.tensor([1.0], dtype=torch.float64)
        r_ik = torch.tensor([1.0], dtype=torch.float64)
        cos_theta = torch.tensor([0.5], dtype=torch.float64)  # 60 degrees

        G_ang = ang_sf(r_ij, r_ik, cos_theta)

        assert G_ang.shape == (1, 4)  # 1 triplet, 4 orders (0-3)

    def test_cutoff_application(self):
        """Test that cutoff is applied to both distances."""
        ang_sf = AngularBasis(ang_order=3, ang_cutoff=2.0)

        # One distance beyond cutoff
        r_ij = torch.tensor([1.0], dtype=torch.float64)
        r_ik = torch.tensor([3.0], dtype=torch.float64)  # Beyond cutoff
        cos_theta = torch.tensor([0.5], dtype=torch.float64)

        G_ang = ang_sf(r_ij, r_ik, cos_theta)

        # Should be all zeros (one distance beyond cutoff)
        assert torch.allclose(G_ang, torch.zeros_like(G_ang))

    def test_angle_extremes(self):
        """Test at extreme angles (0° and 180°)."""
        ang_sf = AngularBasis(ang_order=3, ang_cutoff=2.0)

        r_ij = torch.tensor([1.0, 1.0], dtype=torch.float64)
        r_ik = torch.tensor([1.0, 1.0], dtype=torch.float64)

        # cos(0°) = 1, cos(180°) = -1
        cos_theta = torch.tensor([1.0, -1.0], dtype=torch.float64)

        G_ang = ang_sf(r_ij, r_ik, cos_theta)

        # At cos(θ)=1: T_n(1) = 1 for all n
        # At cos(θ)=-1: T_n(-1) = (-1)^n
        fc_val = ang_sf.cheb.cutoff_function(torch.tensor([1.0]), 2.0)[0]
        expected_0 = fc_val**2 * torch.ones(4, dtype=torch.float64)
        expected_180 = fc_val**2 * torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64)

        assert torch.allclose(G_ang[0], expected_0, atol=1e-10)
        assert torch.allclose(G_ang[1], expected_180, atol=1e-10)

    def test_batch_processing(self):
        """Test batch processing of multiple triplets."""
        ang_sf = AngularBasis(ang_order=3, ang_cutoff=2.0)

        # Multiple triplets
        n_triplets = 5
        r_ij = torch.rand(n_triplets, dtype=torch.float64) * 1.5
        r_ik = torch.rand(n_triplets, dtype=torch.float64) * 1.5
        cos_theta = torch.rand(n_triplets, dtype=torch.float64) * 2 - 1

        G_ang = ang_sf(r_ij, r_ik, cos_theta)

        assert G_ang.shape == (n_triplets, 4)

    def test_cos_theta_clamping(self):
        """Test that cos(θ) is properly clamped to [-1, 1]."""
        ang_sf = AngularBasis(ang_order=3, ang_cutoff=2.0)

        r_ij = torch.tensor([1.0], dtype=torch.float64)
        r_ik = torch.tensor([1.0], dtype=torch.float64)

        # Values outside [-1, 1] (shouldn't happen but test robustness)
        cos_theta = torch.tensor([1.5], dtype=torch.float64)

        # Should not crash
        G_ang = ang_sf(r_ij, r_ik, cos_theta)
        assert G_ang.shape == (1, 4)


class TestNumericalStability:
    """Test numerical stability of implementations."""

    def test_arccos_near_boundaries(self):
        """Test numerical stability near x = ±1."""
        cheb = ChebyshevPolynomials(max_order=10, r_min=-1.0, r_max=1.0)

        # Values very close to ±1
        x = torch.tensor(
            [-0.9999999, -0.999, 0.0, 0.999, 0.9999999], dtype=torch.float64)

        # Should not produce NaN or Inf
        T = cheb(x)
        assert not torch.isnan(T).any()
        assert not torch.isinf(T).any()

    def test_sqrt_term_stability(self):
        """Test stability of sqrt(1-x²) term in derivatives."""
        cheb = ChebyshevPolynomials(max_order=10, r_min=-1.0, r_max=1.0)

        # Values very close to ±1 where sqrt(1-x²) → 0
        x = torch.tensor([-0.9999999, 0.9999999], dtype=torch.float64)

        # Should not produce NaN or Inf
        T, dT_dx = cheb.evaluate_with_derivatives(x)
        assert not torch.isnan(dT_dx).any()
        assert not torch.isinf(dT_dx).any()

    def test_high_order_stability(self):
        """Test stability with high polynomial orders."""
        cheb = ChebyshevPolynomials(max_order=50, r_min=-1.0, r_max=1.0)

        x = torch.linspace(-0.99, 0.99, 10, dtype=torch.float64)

        T = cheb(x)
        assert not torch.isnan(T).any()
        assert not torch.isinf(T).any()

        # Values should be bounded
        # (Chebyshev polynomials ∈ [-1, 1] for x ∈ [-1, 1])
        assert torch.all(torch.abs(T) <= 1.0 + 1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
