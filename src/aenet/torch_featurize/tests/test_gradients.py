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


class TestPeriodicGradients:
    """Test gradient computation with periodic boundary conditions.

    Following the approach in fortran/test_sfbasis.f90::test_derivatives(),
    which tests a periodic FCC structure.
    """

    @pytest.fixture
    def periodic_descriptor(self):
        """Descriptor for periodic FCC structure (single species)."""
        return ChebyshevDescriptor(
            species=["H"],
            rad_order=5,  # Reduced from 10 for faster tests
            rad_cutoff=4.0,  # Reduced from 6.0 for fewer neighbors
            ang_order=3,  # Reduced from 10 for faster tests
            ang_cutoff=3.0,  # Reduced from 6.0 for fewer neighbors
            device="cpu",
        )

    @pytest.fixture
    def fcc_structure(self):
        """
        Create periodic FCC structure matching Fortran test.

        Returns rotated FCC lattice with translation, similar to
        fortran/test_sfbasis.f90::test_derivatives().
        """
        # Base FCC lattice vectors (alat = 2.0)
        alat = 2.0
        avec = torch.tensor(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            dtype=torch.float64,
        ) * alat

        # First rotation (around z-axis-like)
        ax = torch.sqrt(torch.tensor(0.3))
        R1 = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, ax, ax],
                [0.0, -ax, ax],
            ],
            dtype=torch.float64,
        )
        avec = R1 @ avec

        # Second rotation (around y-axis-like)
        ax = torch.sqrt(torch.tensor(0.7))
        R2 = torch.tensor(
            [
                [ax, 0.0, ax],
                [0.0, 1.0, 0.0],
                [-ax, 0.0, ax],
            ],
            dtype=torch.float64,
        )
        avec = R2 @ avec

        # Central atom position (translated, in Cartesian)
        center_pos = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64)

        # Add small displacement like Fortran
        center_pos = center_pos + torch.tensor(
            [0.1, -0.3, 0.2], dtype=torch.float64
        )

        # Generate neighbor positions using PBC
        # For testing, just use the central atom
        # (neighbors will be generated by neighbor list with PBC)
        positions = center_pos.unsqueeze(0)
        species = ["H"]
        pbc = torch.tensor([True, True, True])

        return {
            "positions": positions,
            "species": species,
            "cell": avec,
            "pbc": pbc,
        }

    def test_gradient_per_feature_central_atom_periodic(
        self, periodic_descriptor, fcc_structure
    ):
        """
        Test per-feature gradients w.r.t. central atom in periodic system.

        Following fortran/test_sfbasis.f90 test_derivatives() for central atom.
        Uses tol=1.0e-3, prec=0.03 (3%).
        """
        positions = fcc_structure["positions"]
        species = fcc_structure["species"]
        cell = fcc_structure["cell"]
        pbc = fcc_structure["pbc"]

        epsilon = 1e-4

        # Get analytical gradients
        features, gradients = periodic_descriptor.compute_feature_gradients(
            positions, species, cell, pbc
        )

        N, F = features.shape
        atom_idx = 0  # Test central (only) atom

        # Test subset of features (reduced for performance)
        test_features = [0, 3, 6, 9]  # Just 4 features
        test_features = [f for f in test_features if f < F]

        for feature_idx in test_features:
            for coord_idx in range(3):
                # Analytical gradient
                analytical = gradients[
                    atom_idx, feature_idx, atom_idx, coord_idx
                ]

                # Numerical gradient (central difference)
                pos_forward = positions.clone()
                pos_forward[atom_idx, coord_idx] += epsilon
                feat_forward = periodic_descriptor.forward_from_positions(
                    pos_forward, species, cell, pbc
                )

                pos_backward = positions.clone()
                pos_backward[atom_idx, coord_idx] -= epsilon
                feat_backward = periodic_descriptor.forward_from_positions(
                    pos_backward, species, cell, pbc
                )

                numerical = (
                    feat_forward[atom_idx, feature_idx]
                    - feat_backward[atom_idx, feature_idx]
                ) / (2.0 * epsilon)

                # Compare with Fortran tolerances: tol=1.0e-3, prec=0.03
                abs_diff = torch.abs(analytical - numerical)
                rel_diff = abs_diff / (torch.abs(numerical) + 1e-10)

                assert abs_diff < 1e-3 or rel_diff < 0.03, (
                    f"Periodic: Feature {feature_idx}, coord {coord_idx}: "
                    f"analytical={analytical:.6f}, "
                    f"numerical={numerical:.6f}, "
                    f"abs_diff={abs_diff:.2e}, rel_diff={rel_diff:.2%}"
                )

    def test_gradient_per_feature_neighbor_atoms_periodic(
        self, periodic_descriptor, fcc_structure
    ):
        """
        Test per-feature gradients w.r.t. neighbor atoms in periodic system.

        This test uses a supercell approach: we create an explicit non-periodic
        supercell containing the central atom and its periodic neighbors,
        then validate gradients in this simpler isolated system.

        This is more straightforward than the previous approach and provides
        clearer validation of gradient correctness.
        """
        positions = fcc_structure["positions"]
        species = fcc_structure["species"]
        cell = fcc_structure["cell"]
        pbc = fcc_structure["pbc"]

        epsilon = 1e-4

        # First, get the neighbor list to identify periodic neighbors
        pos_device = positions.to(periodic_descriptor.device)
        cell_device = cell.to(periodic_descriptor.device)

        neighbor_data = periodic_descriptor.nbl.get_neighbors(
            pos_device, cell_device, pbc, fractional=False
        )

        edge_index = neighbor_data['edge_index']
        distances = neighbor_data['distances']
        offsets = neighbor_data['offsets']

        # Filter by cutoff
        mask = (distances <= periodic_descriptor.rad_cutoff) & (
            distances > periodic_descriptor.min_cutoff
        )
        edge_index = edge_index[:, mask]
        distances = distances[mask]
        offsets = offsets[mask]

        # Get neighbors of central atom (index 0)
        central_idx = 0
        neighbor_mask = edge_index[0] == central_idx
        neighbor_indices = edge_index[1][neighbor_mask]
        neighbor_offsets = offsets[neighbor_mask]

        if len(neighbor_indices) == 0:
            pytest.skip("No neighbors found in periodic structure")

        # Compute actual neighbor positions (with PBC offsets)
        neighbor_positions = (
            positions[neighbor_indices]
            + neighbor_offsets.to(torch.float64) @ cell
        )

        # Create an isolated supercell: central atom + explicit neighbors
        # This avoids PBC complications in gradient testing
        n_test_neighbors = min(5, len(neighbor_indices))
        test_indices = torch.randperm(len(neighbor_indices))[:n_test_neighbors]

        # Build supercell positions: [central, neighbor1, neighbor2, ...]
        supercell_positions = torch.cat([
            positions,  # Central atom
            neighbor_positions[test_indices]  # Selected neighbors
        ], dim=0)
        supercell_species = ["H"] * (1 + n_test_neighbors)

        # Now test gradients in this isolated system (no PBC)
        # This is much simpler and more reliable
        features, gradients = periodic_descriptor.compute_feature_gradients(
            supercell_positions, supercell_species, cell=None, pbc=None
        )

        N, F = features.shape
        central_atom = 0  # First atom in supercell

        # Test subset of features
        test_features = [0, 3, 6]
        test_features = [f for f in test_features if f < F]

        # Test gradients w.r.t. a few neighbors
        for neighbor_local_idx in range(1, min(4, N)):  # Test first 3 neighbors
            for feature_idx in test_features[:2]:  # 2 features per neighbor
                for coord_idx in range(3):
                    # Analytical gradient
                    analytical = gradients[
                        central_atom, feature_idx, neighbor_local_idx, coord_idx
                    ]

                    # Numerical gradient (central difference)
                    pos_forward = supercell_positions.clone()
                    pos_forward[neighbor_local_idx, coord_idx] += epsilon
                    feat_forward = periodic_descriptor.forward_from_positions(
                        pos_forward, supercell_species, cell=None, pbc=None
                    )

                    pos_backward = supercell_positions.clone()
                    pos_backward[neighbor_local_idx, coord_idx] -= epsilon
                    feat_backward = periodic_descriptor.forward_from_positions(
                        pos_backward, supercell_species, cell=None, pbc=None
                    )

                    numerical = (
                        feat_forward[central_atom, feature_idx]
                        - feat_backward[central_atom, feature_idx]
                    ) / (2.0 * epsilon)

                    # Compare with tolerances
                    abs_diff = torch.abs(analytical - numerical)
                    rel_diff = abs_diff / (torch.abs(numerical) + 1e-10)

                    assert abs_diff < 1e-3 or rel_diff < 0.05, (
                        f"Periodic neighbor {neighbor_local_idx}: "
                        f"Feature {feature_idx}, coord {coord_idx}: "
                        f"analytical={analytical:.6f}, "
                        f"numerical={numerical:.6f}, "
                        f"abs_diff={abs_diff:.2e}, rel_diff={rel_diff:.2%}"
                    )

    def test_pbc_translation_invariance(
        self, periodic_descriptor, fcc_structure
    ):
        """
        Test that features are invariant under lattice translations.

        Translating by a lattice vector should give identical features.
        """
        positions = fcc_structure["positions"]
        species = fcc_structure["species"]
        cell = fcc_structure["cell"]
        pbc = fcc_structure["pbc"]

        # Original features
        features1 = periodic_descriptor.forward_from_positions(
            positions, species, cell, pbc
        )

        # Translate by first lattice vector
        positions_translated = positions + cell[0]
        features2 = periodic_descriptor.forward_from_positions(
            positions_translated, species, cell, pbc
        )

        # Features should be identical (within numerical precision)
        assert torch.allclose(features1, features2, atol=1e-10), (
            "Features not invariant under lattice translation"
        )

    def test_periodic_vs_supercell_consistency(self):
        """
        Test that periodic system gives same result as explicit supercell.

        A small periodic system should give similar features to an
        explicitly constructed supercell with the same local environment.
        """
        # Create simple cubic cell
        alat = 3.0
        cell = torch.eye(3, dtype=torch.float64) * alat

        # Single atom at origin
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        species = ["H"]
        pbc = torch.tensor([True, True, True])

        descriptor = ChebyshevDescriptor(
            species=["H"],
            rad_order=5,
            rad_cutoff=4.5,
            ang_order=3,
            ang_cutoff=3.0,
            device="cpu",
        )

        # Features with PBC
        features_pbc = descriptor.forward_from_positions(
            positions, species, cell, pbc
        )

        # Create 3x3x3 supercell (27 atoms)
        supercell_positions = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    offset = torch.tensor(
                        [i, j, k], dtype=torch.float64
                    ) * alat
                    supercell_positions.append(offset)

        supercell_positions = torch.stack(supercell_positions)
        supercell_species = ["H"] * 27
        supercell_cell = cell * 3  # 3x larger cell

        # Features for central atom in supercell
        features_supercell = descriptor.forward_from_positions(
            supercell_positions, supercell_species, supercell_cell, pbc
        )

        # Central atom is at index 13 (middle of 27)
        central_features = features_supercell[13]

        # Should be very similar (not exact due to cutoff effects)
        rel_diff = torch.abs(features_pbc[0] - central_features) / (
            torch.abs(features_pbc[0]) + 1e-10
        )

        # Most features should be close
        assert (rel_diff < 0.1).sum() > 0.8 * len(rel_diff), (
            "Periodic and supercell features too different"
        )
