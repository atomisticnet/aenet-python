"""
Tests for training configuration classes.
"""

import numpy as np
import pytest

from ..config import (
    SGD,
    Adam,
    Structure,
    TorchTrainingConfig,
)


class TestStructure:
    """Test Structure dataclass."""

    def test_create_structure(self):
        """Test basic structure creation."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        species = ['O', 'H']
        energy = -10.5

        struct = Structure(
            positions=positions,
            species=species,
            energy=energy
        )

        assert struct.n_atoms == 2
        assert struct.has_forces() is False
        assert struct.is_periodic() is False
        assert np.allclose(struct.positions, positions)

    def test_structure_with_forces(self):
        """Test structure with force data."""
        positions = np.array([[0.0, 0.0, 0.0]])
        species = ['H']
        energy = -1.0
        forces = np.array([[0.1, -0.2, 0.3]])

        struct = Structure(
            positions=positions,
            species=species,
            energy=energy,
            forces=forces
        )

        assert struct.has_forces() is True
        assert np.allclose(struct.forces, forces)

    def test_structure_with_cell(self):
        """Test periodic structure."""
        positions = np.array([[0.0, 0.0, 0.0]])
        species = ['Si']
        energy = -5.0
        cell = np.eye(3) * 5.0
        pbc = np.array([True, True, True])

        struct = Structure(
            positions=positions,
            species=species,
            energy=energy,
            cell=cell,
            pbc=pbc
        )

        assert struct.is_periodic() is True
        assert np.allclose(struct.cell, cell)

    def test_invalid_positions_shape(self):
        """Test validation of positions shape."""
        with pytest.raises(ValueError, match="positions must be"):
            Structure(
                positions=np.array([0.0, 0.0, 0.0]),  # Wrong shape
                species=['H'],
                energy=-1.0
            )

    def test_invalid_species_length(self):
        """Test validation of species length."""
        with pytest.raises(ValueError, match="species length"):
            Structure(
                positions=np.array([[0.0, 0.0, 0.0]]),
                species=['H', 'O'],  # Too many species
                energy=-1.0
            )

    def test_invalid_forces_shape(self):
        """Test validation of forces shape."""
        with pytest.raises(ValueError, match="forces shape"):
            Structure(
                positions=np.array([[0.0, 0.0, 0.0]]),
                species=['H'],
                energy=-1.0,
                forces=np.array([[0.1, 0.2]])  # Wrong shape
            )


class TestTrainingMethod:
    """Test training method configurations."""

    def test_adam_defaults(self):
        """Test Adam with default parameters."""
        adam = Adam()

        assert adam.method_name == "adam"
        assert adam.mu == 0.001
        assert adam.batchsize == 32
        assert adam.weight_decay == 0.0

    def test_adam_custom(self):
        """Test Adam with custom parameters."""
        adam = Adam(mu=0.01, batchsize=64, weight_decay=1e-5)

        assert adam.mu == 0.01
        assert adam.batchsize == 64
        assert adam.weight_decay == 1e-5

    def test_sgd_defaults(self):
        """Test SGD with default parameters."""
        sgd = SGD()

        assert sgd.method_name == "sgd"
        assert sgd.lr == 0.01
        assert sgd.momentum == 0.0

    def test_method_to_params_dict(self):
        """Test conversion to parameter dictionary."""
        adam = Adam(mu=0.005, batchsize=16)
        params = adam.to_params_dict()

        assert 'mu' in params
        assert 'batchsize' in params
        assert params['mu'] == 0.005
        assert params['batchsize'] == 16


class TestTorchTrainingConfig:
    """Test TorchTrainingConfig."""

    def test_default_config(self):
        """Test config with all defaults."""
        config = TorchTrainingConfig()

        assert config.iterations == 100
        assert config.testpercent == 10
        assert config.force_weight == 0.0
        assert config.force_fraction == 1.0
        assert config.force_sampling == 'random'
        assert config.memory_mode == 'gpu'
        assert config.method is not None
        assert isinstance(config.method, Adam)

    def test_custom_config(self):
        """Test config with custom parameters."""
        config = TorchTrainingConfig(
            iterations=500,
            method=SGD(lr=0.1),
            testpercent=20,
            force_weight=0.5,
            force_fraction=0.8,
            force_sampling='fixed',
            memory_mode='cpu'
        )

        assert config.iterations == 500
        assert isinstance(config.method, SGD)
        assert config.testpercent == 20
        assert config.force_weight == 0.5
        assert config.force_fraction == 0.8
        assert config.force_sampling == 'fixed'
        assert config.memory_mode == 'cpu'

    def test_alpha_property(self):
        """Test alpha alias for force_weight."""
        config = TorchTrainingConfig(force_weight=0.3)
        assert config.alpha == 0.3

    def test_batch_size_property(self):
        """Test batch_size property."""
        config = TorchTrainingConfig(method=Adam(batchsize=128))
        assert config.batch_size == 128

    def test_invalid_testpercent(self):
        """Test validation of testpercent."""
        with pytest.raises(ValueError, match="testpercent must be"):
            TorchTrainingConfig(testpercent=150)

        with pytest.raises(ValueError, match="testpercent must be"):
            TorchTrainingConfig(testpercent=-10)

    def test_invalid_force_weight(self):
        """Test validation of force_weight."""
        with pytest.raises(ValueError, match="force_weight must be"):
            TorchTrainingConfig(force_weight=1.5)

        with pytest.raises(ValueError, match="force_weight must be"):
            TorchTrainingConfig(force_weight=-0.1)

    def test_invalid_force_fraction(self):
        """Test validation of force_fraction."""
        with pytest.raises(ValueError, match="force_fraction must be"):
            TorchTrainingConfig(force_fraction=1.2)

    def test_invalid_force_sampling(self):
        """Test validation of force_sampling."""
        with pytest.raises(ValueError, match="force_sampling must be"):
            TorchTrainingConfig(force_sampling='invalid')

    def test_invalid_memory_mode(self):
        """Test validation of memory_mode."""
        with pytest.raises(ValueError, match="memory_mode must be"):
            TorchTrainingConfig(memory_mode='invalid')

    def test_negative_iterations(self):
        """Test validation of iterations."""
        with pytest.raises(ValueError, match="iterations must be"):
            TorchTrainingConfig(iterations=-10)
