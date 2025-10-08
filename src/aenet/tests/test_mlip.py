"""
Unit tests for aenet.mlip module.

"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from aenet.mlip import (ANNPotential, TrainingConfig, Adam, BFGS,
                        EKF, LM, OnlineSD)


class TestANNPotential(unittest.TestCase):
    """Test cases for the ANNPotential class."""

    def setUp(self):
        """Set up test fixtures."""
        self.simple_arch = {
            'Si': [(10, 'tanh'), (10, 'tanh')]
        }
        self.multi_arch = {
            'Li': [(5, 'tanh'), (5, 'tanh')],
            'O': [(8, 'tanh'), (8, 'tanh')],
            'Ti': [(6, 'relu')]
        }

    def test_initialization(self):
        """Test ANNPotential initialization."""
        potential = ANNPotential(self.simple_arch)
        self.assertEqual(potential.arch, self.simple_arch)

    def test_num_types_property(self):
        """Test num_types property."""
        potential = ANNPotential(self.simple_arch)
        self.assertEqual(potential.num_types, 1)

        potential_multi = ANNPotential(self.multi_arch)
        self.assertEqual(potential_multi.num_types, 3)

    def test_atom_types_property(self):
        """Test atom_types property."""
        potential = ANNPotential(self.simple_arch)
        self.assertEqual(potential.atom_types, ['Si'])

        potential_multi = ANNPotential(self.multi_arch)
        self.assertCountEqual(potential_multi.atom_types, ['Li', 'O', 'Ti'])

    def test_train_input_string_basic(self):
        """Test train_input_string with basic parameters."""
        potential = ANNPotential(self.simple_arch)
        config = TrainingConfig(iterations=100, method=BFGS(), testpercent=10)
        input_str = potential.train_input_string(
            trnset_file='data.train',
            config=config
        )

        # Check that all required sections are present
        self.assertIn('TRAININGSET', input_str)
        self.assertIn('TESTPERCENT 10', input_str)
        self.assertIn('ITERATIONS 100', input_str)
        self.assertIn('METHOD', input_str)
        self.assertIn('bfgs', input_str)
        self.assertIn('NETWORKS', input_str)
        self.assertIn('Si', input_str)
        self.assertIn('10:tanh', input_str)

    def test_train_input_string_with_optional_params(self):
        """Test train_input_string with optional parameters."""
        potential = ANNPotential(self.simple_arch)
        config = TrainingConfig(
            iterations=500,
            method=LM(conv=0.001),
            testpercent=15,
            max_energy=10.0,
            timing=True,
            save_energies=True
        )
        input_str = potential.train_input_string(
            trnset_file='data.train',
            config=config
        )

        self.assertIn('MAXENERGY 10.0', input_str)
        self.assertIn('TIMING', input_str)
        self.assertIn('SAVE_ENERGIES', input_str)
        self.assertIn('lm', input_str)
        self.assertIn('conv=0.001', input_str)

    def test_train_input_string_multi_species(self):
        """Test train_input_string with multiple species."""
        potential = ANNPotential(self.multi_arch)
        config = TrainingConfig(iterations=1000)
        input_str = potential.train_input_string(
            trnset_file='data.train',
            config=config
        )

        # Check that all species are present
        self.assertIn('Li', input_str)
        self.assertIn('O', input_str)
        self.assertIn('Ti', input_str)

        # Check layer specifications
        self.assertIn('5:tanh', input_str)
        self.assertIn('8:tanh', input_str)
        self.assertIn('6:relu', input_str)

    def test_train_input_string_relative_path(self):
        """Test that train_input_string creates relative paths correctly."""
        potential = ANNPotential(self.simple_arch)
        config = TrainingConfig(iterations=100)

        # Test with absolute path
        input_str = potential.train_input_string(
            trnset_file='/absolute/path/data.train',
            config=config,
            workdir='/absolute/path'
        )
        self.assertIn('TRAININGSET "data.train"', input_str)

    def test_write_train_input_file(self):
        """Test writing train.in file to disk."""
        potential = ANNPotential(self.simple_arch)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(iterations=100)
            potential.write_train_input_file(
                trnset_file='data.train',
                config=config,
                workdir=tmpdir,
                filename='train.in'
            )

            output_file = os.path.join(tmpdir, 'train.in')
            self.assertTrue(os.path.exists(output_file))

            # Read and verify content
            with open(output_file, 'r') as f:
                content = f.read()
            self.assertIn('ITERATIONS 100', content)
            self.assertIn('NETWORKS', content)

    def test_write_train_input_file_creates_directory(self):
        """Test that write_train_input_file creates directory if needed."""
        potential = ANNPotential(self.simple_arch)

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = os.path.join(tmpdir, 'subdir', 'nested')
            config = TrainingConfig(iterations=100)
            potential.write_train_input_file(
                trnset_file='data.train',
                config=config,
                workdir=workdir
            )

            self.assertTrue(os.path.exists(workdir))
            self.assertTrue(os.path.exists(os.path.join(workdir, 'train.in')))

    def test_write_train_input_file_existing_file_error(self):
        """Test that write_train_input_file raises error for existing file."""
        potential = ANNPotential(self.simple_arch)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file first
            output_file = os.path.join(tmpdir, 'train.in')
            with open(output_file, 'w') as f:
                f.write('existing content')

            # Try to write again - should raise AssertionError
            config = TrainingConfig(iterations=100)
            with self.assertRaises(AssertionError):
                potential.write_train_input_file(
                    trnset_file='data.train',
                    config=config,
                    workdir=tmpdir,
                    filename='train.in'
                )

    @patch('aenet.mlip.TrnSet')
    @patch('aenet.mlip.cfg')
    @patch('aenet.mlip.subprocess.Popen')
    @patch('aenet.mlip.glob.glob')
    @patch('aenet.mlip.shutil.move')
    @patch('aenet.mlip.shutil.rmtree')
    @patch('os.path.exists')
    def test_train_basic(self, mock_exists, mock_rmtree, mock_move, mock_glob,
                         mock_popen, mock_cfg, mock_trnset):
        """Test basic train method execution with mocks."""
        # Setup mocks
        mock_exists.return_value = True
        mock_cfg.read.return_value = {'train_x_path': '/path/to/train.x'}

        # Mock TrnSet context manager
        mock_ts = MagicMock()
        mock_ts.atom_types = ['Si']
        mock_trnset.from_file.return_value.__enter__.return_value = mock_ts
        mock_trnset.from_file.return_value.__exit__.return_value = False

        # Mock subprocess
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, None, 0]  # Running then done
        mock_popen.return_value = mock_proc

        # Mock glob for progress tracking
        mock_glob.return_value = []

        potential = ANNPotential(self.simple_arch)

        with tempfile.TemporaryDirectory() as tmpdir:
            trnset_file = os.path.join(tmpdir, 'data.train')
            # Create dummy training set file
            with open(trnset_file, 'w') as f:
                f.write('dummy')

            # Mock exists to return True for our file
            def exists_side_effect(path):
                return path == trnset_file or path == '/path/to/train.x'
            mock_exists.side_effect = exists_side_effect

            # Run train (will use temp directory)
            config = TrainingConfig(iterations=10)
            potential.train(
                trnset_file=trnset_file,
                config=config
            )

            # Verify train.x was called
            mock_popen.assert_called_once()

    @patch('aenet.mlip.TrnSet')
    @patch('os.path.exists')
    def test_train_missing_trnset_file(self, mock_exists, mock_trnset):
        """Test train raises error for missing training set file."""
        mock_exists.return_value = False

        potential = ANNPotential(self.simple_arch)
        config = TrainingConfig(iterations=10)

        with self.assertRaises(FileNotFoundError) as context:
            potential.train(trnset_file='nonexistent.train', config=config)

        self.assertIn('Training set file not found', str(context.exception))

    @patch('aenet.mlip.cfg')
    @patch('aenet.mlip.TrnSet')
    @patch('os.path.exists')
    def test_train_missing_train_executable(self, mock_exists, mock_trnset,
                                            mock_cfg):
        """Test train raises error for missing train.x executable."""
        # Training set exists but train.x doesn't
        def exists_side_effect(path):
            if 'data.train' in str(path):
                return True
            elif 'train.x' in str(path):
                return False
            return False
        mock_exists.side_effect = exists_side_effect

        mock_cfg.read.return_value = {'train_x_path': '/path/to/train.x'}

        # Mock TrnSet
        mock_ts = MagicMock()
        mock_ts.atom_types = ['Si']
        mock_trnset.from_file.return_value.__enter__.return_value = mock_ts
        mock_trnset.from_file.return_value.__exit__.return_value = False

        potential = ANNPotential(self.simple_arch)
        config = TrainingConfig(iterations=10)

        with self.assertRaises(FileNotFoundError) as context:
            potential.train(trnset_file='data.train', config=config)

        self.assertIn('Cannot find `train.x`', str(context.exception))

    @patch('aenet.mlip.TrnSet')
    @patch('os.path.exists')
    def test_train_incompatible_species(self, mock_exists, mock_trnset):
        """Test train raises error for incompatible species."""
        mock_exists.return_value = True

        # Mock TrnSet with different species
        mock_ts = MagicMock()
        mock_ts.atom_types = ['O', 'H']  # Different from self.simple_arch
        mock_trnset.from_file.return_value.__enter__.return_value = mock_ts
        mock_trnset.from_file.return_value.__exit__.return_value = False

        potential = ANNPotential(self.simple_arch)
        config = TrainingConfig(iterations=10)

        with self.assertRaises(ValueError) as context:
            potential.train(trnset_file='data.train', config=config)

        self.assertIn('Not all species in the training set',
                      str(context.exception))


class TestTrainingMethods(unittest.TestCase):
    """Test cases for training method classes."""

    def test_adam_defaults(self):
        """Test Adam with default parameters."""
        method = Adam()
        self.assertEqual(method.method_name, "adam")
        params = method.to_params_dict()
        self.assertEqual(params['mu'], 0.001)
        self.assertEqual(params['b1'], 0.9)
        self.assertEqual(params['b2'], 0.999)
        self.assertEqual(params['eps'], 1.0e-8)
        self.assertEqual(params['batchsize'], 16)
        self.assertEqual(params['samplesize'], 100)

    def test_adam_custom(self):
        """Test Adam with custom parameters."""
        method = Adam(mu=0.005, batchsize=200)
        params = method.to_params_dict()
        self.assertEqual(params['mu'], 0.005)
        self.assertEqual(params['batchsize'], 200)
        # Other params should still have defaults
        self.assertEqual(params['b1'], 0.9)

    def test_bfgs(self):
        """Test BFGS (no parameters)."""
        method = BFGS()
        self.assertEqual(method.method_name, "bfgs")
        params = method.to_params_dict()
        self.assertEqual(params, {})

    def test_ekf_defaults(self):
        """Test EKF with default parameters."""
        method = EKF()
        self.assertEqual(method.method_name, "ekf")
        params = method.to_params_dict()
        # Check lambda_ is converted to lambda
        self.assertIn('lambda', params)
        self.assertNotIn('lambda_', params)
        self.assertEqual(params['lambda'], 0.99)
        self.assertEqual(params['lambda0'], 0.999)
        self.assertEqual(params['P'], 100.0)
        self.assertEqual(params['mnoise'], 0.0)
        self.assertEqual(params['pnoise'], 1.0e-5)
        self.assertEqual(params['wgmax'], 500)

    def test_ekf_custom(self):
        """Test EKF with custom parameters."""
        method = EKF(lambda_=0.95, P=150.0)
        params = method.to_params_dict()
        self.assertEqual(params['lambda'], 0.95)
        self.assertEqual(params['P'], 150.0)

    def test_lm_defaults(self):
        """Test LM with default parameters."""
        method = LM()
        self.assertEqual(method.method_name, "lm")
        params = method.to_params_dict()
        self.assertEqual(params['batchsize'], 256)
        self.assertEqual(params['learnrate'], 0.1)
        self.assertEqual(params['iter'], 3)
        self.assertEqual(params['conv'], 1e-3)
        self.assertEqual(params['adjust'], 5)

    def test_lm_custom(self):
        """Test LM with custom parameters."""
        method = LM(batchsize=128, learnrate=0.05, conv=1e-4)
        params = method.to_params_dict()
        self.assertEqual(params['batchsize'], 128)
        self.assertEqual(params['learnrate'], 0.05)
        self.assertEqual(params['conv'], 1e-4)

    def test_online_sd_defaults(self):
        """Test OnlineSD with default parameters."""
        method = OnlineSD()
        self.assertEqual(method.method_name, "online_sd")
        params = method.to_params_dict()
        self.assertEqual(params['gamma'], 1.0e-5)
        self.assertEqual(params['alpha'], 0.25)

    def test_online_sd_custom(self):
        """Test OnlineSD with custom parameters."""
        method = OnlineSD(gamma=1e-6, alpha=0.3)
        params = method.to_params_dict()
        self.assertEqual(params['gamma'], 1e-6)
        self.assertEqual(params['alpha'], 0.3)

    def test_train_input_string_default_method(self):
        """Test that Adam is used as default when no method specified."""
        arch = {'Si': [(10, 'tanh')]}
        potential = ANNPotential(arch)
        config = TrainingConfig(iterations=100)
        input_str = potential.train_input_string(
            trnset_file='data.train',
            config=config
        )
        # Should use Adam as default
        self.assertIn('adam', input_str)
        self.assertIn('mu=0.001', input_str)

    def test_train_input_string_all_methods(self):
        """Test train_input_string with each training method."""
        arch = {'Si': [(10, 'tanh')]}
        potential = ANNPotential(arch)

        # Test BFGS
        config = TrainingConfig(iterations=100, method=BFGS())
        input_str = potential.train_input_string(
            trnset_file='data.train',
            config=config
        )
        self.assertIn('bfgs', input_str)
        # BFGS should have no parameters on METHOD line
        self.assertIn('METHOD\nbfgs\n', input_str)

        # Test Adam
        config = TrainingConfig(iterations=100, method=Adam(mu=0.002))
        input_str = potential.train_input_string(
            trnset_file='data.train',
            config=config
        )
        self.assertIn('adam', input_str)
        self.assertIn('mu=0.002', input_str)

        # Test EKF
        config = TrainingConfig(iterations=100, method=EKF(lambda_=0.95))
        input_str = potential.train_input_string(
            trnset_file='data.train',
            config=config
        )
        self.assertIn('ekf', input_str)
        self.assertIn('lambda=0.95', input_str)  # Note: lambda not lambda_

        # Test LM
        config = TrainingConfig(iterations=100, method=LM(batchsize=128))
        input_str = potential.train_input_string(
            trnset_file='data.train',
            config=config
        )
        self.assertIn('lm', input_str)
        self.assertIn('batchsize=128', input_str)

        # Test OnlineSD
        config = TrainingConfig(iterations=100, method=OnlineSD(gamma=1e-6))
        input_str = potential.train_input_string(
            trnset_file='data.train',
            config=config
        )
        self.assertIn('online_sd', input_str)
        self.assertIn('gamma=1e-06', input_str)


class TestTrainingConfig(unittest.TestCase):
    """Test cases for TrainingConfig validation."""

    def test_training_config_defaults(self):
        """Test TrainingConfig with default values."""
        config = TrainingConfig()
        self.assertEqual(config.iterations, 0)
        self.assertIsInstance(config.method, Adam)
        self.assertEqual(config.testpercent, 0)
        self.assertIsNone(config.max_energy)
        self.assertIsNone(config.sampling)
        self.assertFalse(config.timing)
        self.assertFalse(config.save_energies)

    def test_training_config_custom(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            iterations=1000,
            method=BFGS(),
            testpercent=10,
            max_energy=100.0,
            sampling='random',
            timing=True,
            save_energies=True
        )
        self.assertEqual(config.iterations, 1000)
        self.assertIsInstance(config.method, BFGS)
        self.assertEqual(config.testpercent, 10)
        self.assertEqual(config.max_energy, 100.0)
        self.assertEqual(config.sampling, 'random')
        self.assertTrue(config.timing)
        self.assertTrue(config.save_energies)

    def test_training_config_invalid_testpercent(self):
        """Test TrainingConfig rejects invalid testpercent."""
        with self.assertRaises(ValueError) as context:
            TrainingConfig(testpercent=-1)
        self.assertIn('testpercent must be 0-100', str(context.exception))

        with self.assertRaises(ValueError) as context:
            TrainingConfig(testpercent=101)
        self.assertIn('testpercent must be 0-100', str(context.exception))

    def test_training_config_invalid_sampling(self):
        """Test TrainingConfig rejects invalid sampling."""
        with self.assertRaises(ValueError) as context:
            TrainingConfig(sampling='invalid')
        self.assertIn('sampling must be one of', str(context.exception))

    def test_training_config_invalid_iterations(self):
        """Test TrainingConfig rejects invalid iterations."""
        with self.assertRaises(ValueError) as context:
            TrainingConfig(iterations=-10)
        self.assertIn('iterations must be >= 0', str(context.exception))

    def test_training_config_valid_sampling_values(self):
        """Test TrainingConfig accepts all valid sampling values."""
        valid_sampling = ['sequential', 'random', 'weighted', 'energy']
        for sampling in valid_sampling:
            config = TrainingConfig(sampling=sampling)
            self.assertEqual(config.sampling, sampling)


if __name__ == '__main__':
    unittest.main()
