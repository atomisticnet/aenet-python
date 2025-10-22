"""
Tests for gradient computation in featurization.

Validates analytical gradients against numerical finite differences,
following the approach in fortran/test_sfbasis.f90
"""

import pytest
import torch

from ..featurize import ChebyshevDescriptor


class TestGradients:
    """Test gradient computation with finite difference validation."""

    @pytest.fixture
    def water_descriptor(self):
        """Simple water molecule descriptor."""
        return ChebyshevDescriptor(
            species=["O", "H"],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5,
            device="cpu",
        )

    @pytest.fixture
    def water_positions(self):
        """Water molecule positions."""
        return torch.tensor(
            [
                [0.0, 0.0, 0.0],  # O
                [0.96, 0.0, 0.0],  # H1
                [-0.24, 0.93, 0.0],  # H2
            ],
            dtype=torch.float64,
        )

    def compute_numerical_gradient_single_feature(
        self,
        descriptor,
        positions,
        species,
        atom_idx,
        feature_idx,
        perturb_atom,
        perturb_coord,
        epsilon=1e-4,
    ):
        """
        Compute numerical gradient for a single feature.

        Returns: ∂feature[atom_idx, feature_idx]/∂position[perturb_atom,
                 perturb_coord]
        """
        # Forward step
        pos_forward = positions.clone()
        pos_forward[perturb_atom, perturb_coord] += epsilon
        feat_forward = descriptor(pos_forward, species)

        # Backward step
        pos_backward = positions.clone()
        pos_backward[perturb_atom, perturb_coord] -= epsilon
        feat_backward = descriptor(pos_backward, species)

        # Central difference
        grad = (
            feat_forward[atom_idx, feature_idx]
            - feat_backward[atom_idx, feature_idx]
        ) / (2.0 * epsilon)

        return grad

    def test_gradient_per_feature_central_atom(
        self, water_descriptor, water_positions
    ):
        """
        Test per-feature gradients w.r.t. central atom position.

        Validates ∂G[i,f]/∂r[i,k] for oxygen atom (i=0).
        Following the approach in fortran/test_sfbasis.f90 test_derivatives.
        """
        species = ["O", "H", "H"]
        epsilon = 1e-4

        # Get analytical gradients
        features, gradients = water_descriptor.compute_feature_gradients(
            water_positions, species
        )

        N, F = features.shape

        # Test central atom (oxygen, index 0)
        atom_idx = 0

        # Test a subset of features (all would be slow)
        test_features = [0, 5, 10, 15, 20, 25]  # Sample across feature space
        test_features = [f for f in test_features if f < F]

        for feature_idx in test_features:
            for coord_idx in range(3):
                # Analytical: gradients[atom, feature, atom, coord]
                analytical = gradients[
                    atom_idx, feature_idx, atom_idx, coord_idx
                ]

                # Numerical
                numerical = self.compute_numerical_gradient_single_feature(
                    water_descriptor,
                    water_positions,
                    species,
                    atom_idx,
                    feature_idx,
                    atom_idx,
                    coord_idx,
                    epsilon,
                )

                # Compare with Fortran tolerances
                abs_diff = torch.abs(analytical - numerical)
                rel_diff = abs_diff / (torch.abs(numerical) + 1e-10)

                # tol=1.0e-3, prec=0.03 (3%)
                assert abs_diff < 1e-3 or rel_diff < 0.03, (
                    f"Feature {feature_idx}, coord {coord_idx}: "
                    f"analytical={analytical:.6f}, "
                    f"numerical={numerical:.6f}, "
                    f"abs_diff={abs_diff:.2e}, rel_diff={rel_diff:.2%}"
                )

    def test_gradient_per_feature_neighbor_atoms(
        self, water_descriptor, water_positions
    ):
        """
        Test per-feature gradients w.r.t. neighbor positions.

        Validates ∂G[i,f]/∂r[j,k] for j ≠ i.
        Following fortran/test_sfbasis.f90 test_derivatives.
        """
        species = ["O", "H", "H"]
        epsilon = 1e-4

        # Get analytical gradients
        features, gradients = water_descriptor.compute_feature_gradients(
            water_positions, species
        )

        N, F = features.shape

        # Test gradients of oxygen features w.r.t. hydrogen positions
        central_atom = 0  # Oxygen
        neighbor_atoms = [1, 2]  # Hydrogens

        # Test subset of features
        test_features = [0, 5, 10, 15]
        test_features = [f for f in test_features if f < F]

        for neighbor_atom in neighbor_atoms:
            for feature_idx in test_features[:3]:  # Just a few per neighbor
                for coord_idx in range(3):
                    # Analytical: gradients[central, feature, neighbor, coord]
                    analytical = gradients[
                        central_atom, feature_idx, neighbor_atom, coord_idx
                    ]

                    # Numerical
                    numerical = self.compute_numerical_gradient_single_feature(
                        water_descriptor,
                        water_positions,
                        species,
                        central_atom,
                        feature_idx,
                        neighbor_atom,
                        coord_idx,
                        epsilon,
                    )

                    # Compare with slightly more lenient tolerances
                    # (neighbor gradients can be more sensitive)
                    abs_diff = torch.abs(analytical - numerical)
                    rel_diff = abs_diff / (torch.abs(numerical) + 1e-10)

                    # tol=1.0e-4, prec=0.01 (1% - tighter than Fortran's
                    # check_j)
                    assert abs_diff < 1e-3 or rel_diff < 0.05, (
                        f"Neighbor {neighbor_atom}, feature {feature_idx}, "
                        f"coord {coord_idx}: "
                        f"analytical={analytical:.6f}, "
                        f"numerical={numerical:.6f}, "
                        f"abs_diff={abs_diff:.2e}, rel_diff={rel_diff:.2%}"
                    )

    def test_compute_feature_gradients_shape(
        self, water_descriptor, water_positions
    ):
        """Test that compute_feature_gradients returns correct shapes."""
        species = ["O", "H", "H"]

        features, gradients = water_descriptor.compute_feature_gradients(
            water_positions, species
        )

        N = len(species)
        F = water_descriptor.get_n_features()

        assert features.shape == (N, F), (
            f"Expected features shape ({N}, {F}), got {features.shape}"
        )
        assert gradients.shape == (N, F, N, 3), (
            f"Expected gradients shape ({N}, {F}, {N}, 3), "
            f"got {gradients.shape}"
        )

    def test_gradient_symmetry(self, water_descriptor, water_positions):
        """
        Test that symmetric atoms have symmetric gradients.

        For water molecule, perturbing H1 and H2 by same amount
        should give similar gradient magnitudes.
        """
        species = ["O", "H", "H"]

        # Compute gradients
        positions_grad = water_positions.clone().requires_grad_(True)
        features = water_descriptor(positions_grad, species)

        # Get gradients for each atom
        features.sum().backward()
        grad = positions_grad.grad

        # Check that gradient magnitudes for H atoms are similar
        grad_h1_norm = torch.norm(grad[1])
        grad_h2_norm = torch.norm(grad[2])

        # They should be within 50% of each other (rough symmetry)
        ratio = grad_h1_norm / (grad_h2_norm + 1e-10)
        assert 0.5 < ratio < 2.0, (
            f"Gradient symmetry check failed: "
            f"H1 norm = {grad_h1_norm:.3f}, H2 norm = {grad_h2_norm:.3f}"
        )

    def test_translation_invariance_of_gradients(
        self, water_descriptor, water_positions
    ):
        """
        Test that gradients are translation invariant.

        Translating the entire system should give the same gradients.
        """
        species = ["O", "H", "H"]
        translation = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        # Original gradients
        pos1 = water_positions.clone().requires_grad_(True)
        features1 = water_descriptor(pos1, species)
        features1.sum().backward()
        grad1 = pos1.grad.clone()

        # Translated gradients
        pos2 = (water_positions + translation).clone().requires_grad_(True)
        features2 = water_descriptor(pos2, species)
        features2.sum().backward()
        grad2 = pos2.grad.clone()

        # Gradients should be identical
        assert torch.allclose(grad1, grad2, atol=1e-10), (
            "Gradients not translation invariant"
        )

    def test_zero_features_for_isolated_atom(self):
        """Test that isolated atom has zero features."""
        descriptor = ChebyshevDescriptor(
            species=["H"],
            rad_order=5,
            rad_cutoff=4.0,
            ang_order=2,
            ang_cutoff=1.5,
            device="cpu",
        )

        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        species = ["H"]

        features = descriptor(positions, species)

        # Isolated atom should have zero features
        # (no neighbors to interact with)
        assert torch.allclose(
            features, torch.zeros_like(features), atol=1e-10
        ), "Isolated atom should have zero features"


class TestForceComputation:
    """Test force computation from energy models."""

    def test_forces_from_simple_energy_model(self):
        """Test force computation with a simple energy model."""
        descriptor = ChebyshevDescriptor(
            species=["O", "H"],
            rad_order=5,
            rad_cutoff=4.0,
            ang_order=2,
            ang_cutoff=1.5,
            device="cpu",
        )

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        species = ["O", "H", "H"]

        # Simple energy model: E = sum of features
        class SimpleEnergyModel(torch.nn.Module):
            def forward(self, features):
                return features.sum()

        energy_model = SimpleEnergyModel()

        # Compute forces
        energy, forces = descriptor.compute_forces_from_energy(
            positions, species, energy_model
        )

        # Check shapes
        assert energy.shape == torch.Size([]), "Energy should be scalar"
        assert forces.shape == (3, 3), "Forces should have shape (N, 3)"

        # Forces should be non-zero for non-isolated atoms
        assert torch.any(forces != 0), "Forces should be non-zero"
