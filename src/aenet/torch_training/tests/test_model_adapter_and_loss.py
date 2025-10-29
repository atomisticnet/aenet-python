"""
Tests for EnergyModelAdapter and loss functions.

These tests avoid heavy dependencies by using small dummy models and
descriptors with synthetic gradients to validate correctness and API
integration.
"""

import numpy as np
import torch
import torch.nn as nn

from ..loss import (
    compute_combined_loss,
    compute_energy_loss,
    compute_force_loss,
)
from ..model_adapter import EnergyModelAdapter


class DummyNetAtom(nn.Module):
    """
    Minimal NetAtom-like structure:

    - functions: per-species nn.Sequential mapping (F,) -> (1,)
    - device attribute
    """

    def __init__(self, in_features_per_species,
                 device="cpu", dtype=torch.float64):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.functions = nn.ModuleList()
        for F in in_features_per_species:
            seq = nn.Sequential(nn.Linear(F, 1, bias=True).to(dtype))
            self.functions.append(seq)
        self.to(device)


class DummyDescriptor:
    """
    Descriptor that returns pre-defined features and feature gradients.
    Used to test compute_force_loss and compute_combined_loss without
    invoking actual featurization or neighbor search.
    """

    def __init__(self, features: torch.Tensor, grad_features: torch.Tensor):
        """
        Parameters
        ----------
        features : (N, F) torch.Tensor
        grad_features : (N, F, N, 3) torch.Tensor
        """
        self.features = features
        self.grad_features = grad_features

    def compute_feature_gradients(self, positions, species,
                                  cell=None, pbc=None):
        # Ignore inputs; return stored tensors
        return self.features.clone(), self.grad_features.clone()

    def compute_feature_gradients_from_neighbor_info(
        self, positions, species, neighbor_indices, neighbor_vectors
    ):
        # Same behavior; we test that neighbor_info and
        # non-neighbor_info paths match
        return self.features.clone(), self.grad_features.clone()


class TestEnergyModelAdapter:
    def test_forward_atomic_two_species_linear(self):
        """
        Build a dummy per-species linear model and verify adapter returns
        correct per-atom energies in original order.
        """
        torch.manual_seed(0)
        dtype = torch.float64
        device = "cpu"

        # Two species with feature size 3 each
        in_features_per_species = [3, 3]
        net = DummyNetAtom(in_features_per_species, device=device, dtype=dtype)

        # Initialize weights/biases for reproducibility
        with torch.no_grad():
            # species 0
            net.functions[0][0].weight.copy_(
                torch.tensor([[0.1, -0.2, 0.3]], dtype=dtype))
            net.functions[0][0].bias.copy_(torch.tensor([0.5], dtype=dtype))
            # species 1
            net.functions[1][0].weight.copy_(
                torch.tensor([[0.4, 0.0, -0.1]], dtype=dtype))
            net.functions[1][0].bias.copy_(torch.tensor([-0.2], dtype=dtype))

        adapter = EnergyModelAdapter(net, n_species=2)

        # 5 atoms, species: 0,1,0,1,0
        species_indices = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
        features = torch.tensor(
            [
                [1.0, 2.0, 3.0],    # sp0
                [0.5, -1.0, 2.0],   # sp1
                [0.0, 0.0, 1.0],    # sp0
                [3.0, 1.0, 0.0],    # sp1
                [-1.0, 2.0, -2.0],  # sp0
            ],
            dtype=dtype,
        )

        # Manually compute expected energies
        def sp0(x):  # 0.1*x0 - 0.2*x1 + 0.3*x2 + 0.5
            return 0.1 * x[0] - 0.2 * x[1] + 0.3 * x[2] + 0.5

        def sp1(x):  # 0.4*x0 + 0.0*x1 - 0.1*x2 - 0.2
            return 0.4 * x[0] + 0.0 * x[1] - 0.1 * x[2] - 0.2

        expected = torch.tensor(
            [
                sp0(features[0]),
                sp1(features[1]),
                sp0(features[2]),
                sp1(features[3]),
                sp0(features[4]),
            ],
            dtype=dtype,
        )

        E_atomic = adapter(features, species_indices)
        assert torch.allclose(E_atomic, expected, atol=1e-12)


class TestEnergyLoss:
    def test_compute_energy_loss_two_structures(self):
        """
        Two small structures concatenated; verify per-atom to per-structure
        aggregation and RMSE ~ 0 when reference matches prediction.
        """
        dtype = torch.float64
        device = "cpu"
        torch.manual_seed(0)

        # Two species; features size 2 each
        in_features_per_species = [2, 2]
        net = DummyNetAtom(in_features_per_species, device=device, dtype=dtype)
        # Simple weights for determinism
        with torch.no_grad():
            net.functions[0][0].weight.copy_(
                torch.tensor([[1.0, 0.0]], dtype=dtype))
            net.functions[0][0].bias.copy_(torch.tensor([0.0], dtype=dtype))
            net.functions[1][0].weight.copy_(
                torch.tensor([[0.0, 1.0]], dtype=dtype))
            net.functions[1][0].bias.copy_(torch.tensor([0.0], dtype=dtype))

        adapter = EnergyModelAdapter(net, n_species=2)

        # Structure A: 3 atoms; Structure B: 2 atoms (total N=5)
        n_atoms = torch.tensor([3, 2], dtype=dtype)
        species_indices = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
        features = torch.tensor(
            [
                [1.0, 2.0],  # sp0 -> energy = 1.0
                [3.0, 4.0],  # sp1 -> energy = 4.0
                [5.0, 6.0],  # sp0 -> energy = 5.0
                [7.0, 8.0],  # sp1 -> energy = 8.0
                [9.0, 0.0],  # sp0 -> energy = 9.0
            ],
            dtype=dtype,
        )

        # Predicted per-structure energies: A=(1+4+5)=10, B=(8+9)=17
        energy_ref = torch.tensor([10.0, 17.0], dtype=dtype)

        loss, energy_pred = compute_energy_loss(
            features=features,
            energy_ref=energy_ref,
            n_atoms=n_atoms,
            network=adapter,
            species_indices=species_indices,
            E_shift=0.0,
            E_scaling=1.0,
        )

        assert torch.allclose(energy_pred, energy_ref, atol=1e-12)
        assert loss.item() < 1e-12


class TestForceLoss:
    def _make_dummy_descriptor(self, N=4, F=3, seed=0):
        torch.manual_seed(seed)
        dtype = torch.float64
        device = "cpu"
        # Synthetic features
        features = torch.randn(N, F, dtype=dtype, device=device)
        # Synthetic gradients (N, F, N, 3)
        grad_features = torch.randn(N, F, N, 3, dtype=dtype, device=device)
        return DummyDescriptor(features, grad_features)

    def _make_simple_adapter(self, F=3):
        dtype = torch.float64
        device = "cpu"
        in_features_per_species = [F, F]  # 2 species
        net = DummyNetAtom(in_features_per_species, device=device, dtype=dtype)
        # identity-like: species 0 sums first feature; species 1 sums second
        with torch.no_grad():
            net.functions[0][0].weight.zero_()
            net.functions[0][0].weight[0, 0] = 1.0
            net.functions[0][0].bias.zero_()
            net.functions[1][0].weight.zero_()
            net.functions[1][0].weight[0, 1] = 1.0
            net.functions[1][0].bias.zero_()
        return EnergyModelAdapter(net, n_species=2)

    def test_compute_force_loss_neighbor_info_and_chunking(self):
        """
        Validate force loss path using dummy descriptor with synthetic
        gradients. Ensure neighbor_info path equals recomputed path and
        chunking equals non-chunked results.
        """
        N = 5
        F = 4
        dtype = torch.float64

        descriptor = self._make_dummy_descriptor(N=N, F=F, seed=1)
        adapter = self._make_simple_adapter(F=F)

        # Positions/species (unused by DummyDescriptor but required by API)
        positions = torch.zeros(N, 3, dtype=dtype)
        species = ["A" if i % 2 == 0 else "B" for i in range(N)]
        species_indices = torch.tensor(
            [0 if s == "A" else 1 for s in species], dtype=torch.long)

        # Reference forces: use zeros to test consistency
        # (we only validate equality paths)
        forces_ref = torch.zeros(N, 3, dtype=dtype)

        # No neighbor_info path
        loss1, forces1 = compute_force_loss(
            positions=positions,
            species=species,
            forces_ref=forces_ref,
            descriptor=descriptor,
            network=adapter,
            species_indices=species_indices,
            E_scaling=1.0,
            neighbor_info=None,
            chunk_size=None,
        )

        # Provide dummy neighbor_info (content not used by DummyDescriptor)
        neighbor_info = {
            "neighbor_lists": [np.array([], dtype=np.int64) for _ in range(N)],
            "neighbor_vectors": [np.zeros((0, 3)) for _ in range(N)],
        }

        loss2, forces2 = compute_force_loss(
            positions=positions,
            species=species,
            forces_ref=forces_ref,
            descriptor=descriptor,
            network=adapter,
            species_indices=species_indices,
            E_scaling=1.0,
            neighbor_info=neighbor_info,
            chunk_size=None,
        )

        assert torch.allclose(forces1, forces2, atol=1e-12)
        assert torch.allclose(loss1, loss2, atol=1e-12)

        # Chunked vs non-chunked
        loss3, forces3 = compute_force_loss(
            positions=positions,
            species=species,
            forces_ref=forces_ref,
            descriptor=descriptor,
            network=adapter,
            species_indices=species_indices,
            E_scaling=1.0,
            neighbor_info=neighbor_info,
            chunk_size=2,
        )

        assert torch.allclose(forces1, forces3, atol=1e-12)
        assert torch.allclose(loss1, loss3, atol=1e-12)


class TestCombinedLoss:
    def test_combined_loss_alpha_weighting(self):
        """
        Verify combined loss is (1-alpha)*E + alpha*F.
        """
        dtype = torch.float64
        # Fabricate small tensors
        N = 3
        F = 2

        # Network that returns per-atom energy = sum of features
        class SumAdapter(nn.Module):
            def forward(self, features, species_indices):
                return features.sum(dim=1)

        adapter = SumAdapter()

        # Features: structure 1 has 2 atoms; structure 2 has 1 atom
        n_atoms = torch.tensor([2.0, 1.0], dtype=dtype)
        species_indices = torch.tensor([0, 0, 0], dtype=torch.long)
        features = torch.tensor(
            [
                [1.0, 2.0],  # E=3
                [0.0, 1.0],  # E=1
                [2.0, 2.0],  # E=4
            ],
            dtype=dtype,
        )
        energy_ref = torch.tensor([4.0, 4.0], dtype=dtype)

        # Dummy descriptor that returns gradients contracting to zero forces
        zero_grad = torch.zeros(N, F, N, 3, dtype=dtype)
        descriptor = DummyDescriptor(features=features,
                                     grad_features=zero_grad)
        positions = torch.zeros(N, 3, dtype=dtype)
        species = ["X"] * N
        forces_ref = torch.zeros(N, 3, dtype=dtype)

        # alpha = 0 (energy only)
        combined0, metrics0 = compute_combined_loss(
            features=features,
            energy_ref=energy_ref,
            n_atoms=n_atoms,
            positions=positions,
            species=species,
            forces_ref=forces_ref,
            network=adapter,
            species_indices=species_indices,
            descriptor=descriptor,
            alpha=0.0,
            E_shift=0.0,
            E_scaling=1.0,
            use_forces=True,  # Should be ignored since alpha=0
        )
        assert metrics0["force_loss"] is None
        assert metrics0["energy_loss"].item() < 1e-12
        assert torch.allclose(combined0, metrics0["energy_loss"], atol=1e-12)

        # alpha = 1 (force only)
        combined1, metrics1 = compute_combined_loss(
            features=features,
            energy_ref=energy_ref,
            n_atoms=n_atoms,
            positions=positions,
            species=species,
            forces_ref=forces_ref,
            network=adapter,
            species_indices=species_indices,
            descriptor=descriptor,
            alpha=1.0,
            E_shift=0.0,
            E_scaling=1.0,
            use_forces=True,
        )
        # Forces are zero; force_loss should be zero
        assert metrics1["force_loss"].item() < 1e-12
        assert torch.allclose(combined1, metrics1["force_loss"], atol=1e-12)

        # alpha = 0.25 (weighted)
        combineda, metricsa = compute_combined_loss(
            features=features,
            energy_ref=energy_ref,
            n_atoms=n_atoms,
            positions=positions,
            species=species,
            forces_ref=forces_ref,
            network=adapter,
            species_indices=species_indices,
            descriptor=descriptor,
            alpha=0.25,
            E_shift=0.0,
            E_scaling=1.0,
            use_forces=True,
        )
        expected = ((1.0 - 0.25) * metricsa["energy_loss"]
                    + 0.25 * metricsa["force_loss"])
        assert torch.allclose(combineda, expected, atol=1e-12)
