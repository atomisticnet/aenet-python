"""
Unit tests for aenet.mlip module.

"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

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

    def test_train_basic(self):
        """Test TrainOut parsing with real train.out file."""
        from aenet.io.train import TrainOut

        # Use real train.out file from test data
        test_dir = os.path.dirname(__file__)
        data_dir = os.path.join(test_dir, 'data')
        train_out_file = os.path.join(data_dir, 'train.out')

        # Parse the training output
        result = TrainOut(path=train_out_file)

        # Verify result is a TrainOut object
        self.assertIsInstance(result, TrainOut)

        # Verify the parsed data (the file has 11 epochs: 0-10)
        self.assertEqual(len(result.errors), 11)

        # Check column names
        expected_columns = ['MAE_train', 'RMSE_train', 'MAE_test', 'RMSE_test']
        self.assertListEqual(list(result.errors.columns), expected_columns)

        # Verify some expected values from the file
        # Final epoch (10) values from the train.out file
        self.assertAlmostEqual(
            result.errors['MAE_train'].iloc[-1], 0.5355827, places=5)
        self.assertAlmostEqual(
            result.errors['RMSE_train'].iloc[-1], 0.6168381, places=5)
        self.assertAlmostEqual(
            result.errors['MAE_test'].iloc[-1], 0.6015781, places=5)
        self.assertAlmostEqual(
            result.errors['RMSE_test'].iloc[-1], 0.6244984, places=5)

        # Verify stats
        stats = result.stats
        self.assertIn('final_MAE_train', stats)
        self.assertIn('final_RMSE_test', stats)
        self.assertIn('min_RMSE_test', stats)
        self.assertAlmostEqual(stats['final_RMSE_test'], 0.6244984, places=5)

    @patch('aenet.mlip.TrnSet')
    @patch('os.path.exists')
    def test_train_missing_trnset_file(self, mock_exists, mock_trnset):
        """Test train raises error for missing training set file."""
        mock_exists.return_value = False

        potential = ANNPotential(self.simple_arch)
        config = TrainingConfig(iterations=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with self.assertRaises(FileNotFoundError) as context:
                    potential.train(
                        trnset_file='nonexistent.train', config=config)
            finally:
                os.chdir(cwd)

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

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with self.assertRaises(FileNotFoundError) as context:
                    potential.train(trnset_file='data.train', config=config)
            finally:
                os.chdir(cwd)

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

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with self.assertRaises(ValueError) as context:
                    potential.train(trnset_file='data.train', config=config)
            finally:
                os.chdir(cwd)

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


class TestPredictionConfig(unittest.TestCase):
    """Test cases for PredictionConfig validation."""

    def test_prediction_config_defaults(self):
        """Test PredictionConfig with default values."""
        from aenet.mlip import PredictionConfig
        config = PredictionConfig()
        self.assertIsNone(config.potential_paths)
        self.assertIsNone(config.potential_format)
        self.assertFalse(config.timing)
        self.assertFalse(config.print_atomic_energies)
        self.assertFalse(config.debug)
        self.assertEqual(config.verbosity, 1)

    def test_prediction_config_custom(self):
        """Test PredictionConfig with custom values."""
        from aenet.mlip import PredictionConfig
        config = PredictionConfig(
            potential_paths={'Ti': 'Ti.nn', 'O': 'O.nn'},
            potential_format='ascii',
            timing=True,
            print_atomic_energies=True,
            debug=True,
            verbosity=2
        )
        self.assertEqual(config.potential_paths, {'Ti': 'Ti.nn', 'O': 'O.nn'})
        self.assertEqual(config.potential_format, 'ascii')
        self.assertTrue(config.timing)
        self.assertTrue(config.print_atomic_energies)
        self.assertTrue(config.debug)
        self.assertEqual(config.verbosity, 2)

    def test_prediction_config_invalid_verbosity(self):
        """Test PredictionConfig rejects invalid verbosity."""
        from aenet.mlip import PredictionConfig
        with self.assertRaises(ValueError) as context:
            PredictionConfig(verbosity=3)
        self.assertIn('verbosity must be 0, 1, or 2', str(context.exception))

        with self.assertRaises(ValueError) as context:
            PredictionConfig(verbosity=-1)
        self.assertIn('verbosity must be 0, 1, or 2', str(context.exception))

    def test_prediction_config_invalid_format(self):
        """Test PredictionConfig rejects invalid potential format."""
        from aenet.mlip import PredictionConfig
        with self.assertRaises(ValueError) as context:
            PredictionConfig(potential_format='invalid')
        self.assertIn("potential_format must be 'ascii', 'ASCII', or None",
                      str(context.exception))


class TestANNPotentialPrediction(unittest.TestCase):
    """Test cases for ANNPotential prediction functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.simple_arch = {
            'Ti': [(10, 'tanh'), (10, 'tanh')],
            'O': [(10, 'tanh'), (10, 'tanh')]
        }

    def test_from_files_factory(self):
        """Test ANNPotential.from_files() factory method."""
        test_dir = os.path.dirname(__file__)
        data_dir = os.path.join(test_dir, 'data')

        potential_paths = {
            'Ti': os.path.join(data_dir, 'Ti.nn.ascii'),
            'O': os.path.join(data_dir, 'O.nn.ascii')
        }
        potential = ANNPotential.from_files(
            potential_paths,
            potential_format='ascii'
        )

        self.assertEqual(potential._potential_paths, potential_paths)
        self.assertEqual(potential._potential_format, 'ascii')
        self.assertTrue(potential._from_files)
        self.assertCountEqual(potential.atom_types, ['Ti', 'O'])

    def test_from_files_missing_file_error(self):
        """Test from_files() raises error for missing files."""
        potential_paths = {
            'Ti': 'nonexistent_Ti.nn',
            'O': 'nonexistent_O.nn'
        }

        with self.assertRaises(FileNotFoundError) as context:
            ANNPotential.from_files(potential_paths)

        # Check that the error message lists missing files
        error_msg = str(context.exception)
        self.assertIn('Potential file(s) not found', error_msg)
        self.assertIn('Ti: nonexistent_Ti.nn', error_msg)
        self.assertIn('O: nonexistent_O.nn', error_msg)

    def test_from_files_partial_missing_files(self):
        """Test from_files() with some files missing."""
        test_dir = os.path.dirname(__file__)
        data_dir = os.path.join(test_dir, 'data')

        potential_paths = {
            'Ti': os.path.join(data_dir, 'Ti.nn.ascii'),  # exists
            'O': 'nonexistent_O.nn'  # doesn't exist
        }

        with self.assertRaises(FileNotFoundError) as context:
            ANNPotential.from_files(potential_paths)

        error_msg = str(context.exception)
        self.assertIn('O: nonexistent_O.nn', error_msg)
        # Ti file exists, so it shouldn't be in the error
        self.assertNotIn('Ti:', error_msg)

    def test_predict_input_string_basic(self):
        """Test predict_input_string with basic parameters."""
        from aenet.mlip import PredictionConfig

        potential = ANNPotential(self.simple_arch)
        potential._potential_paths = {'Ti': 'Ti.nn', 'O': 'O.nn'}

        xsf_files = ['structure1.xsf', 'structure2.xsf']
        config = PredictionConfig(verbosity=1)

        input_str = potential.predict_input_string(
            xsf_files=xsf_files,
            eval_forces=False,
            config=config
        )

        # Check required sections
        self.assertIn('TYPES', input_str)
        self.assertIn('2', input_str)  # 2 types
        self.assertIn('Ti', input_str)
        self.assertIn('O', input_str)
        self.assertIn('NETWORKS', input_str)
        self.assertIn('VERBOSITY 1', input_str)
        self.assertIn('FILES', input_str)
        self.assertIn('2', input_str)  # 2 files
        self.assertIn('structure1.xsf', input_str)
        self.assertIn('structure2.xsf', input_str)

    def test_predict_input_string_with_forces(self):
        """Test predict_input_string with forces enabled."""
        from aenet.mlip import PredictionConfig

        potential = ANNPotential(self.simple_arch)
        potential._potential_paths = {'Ti': 'Ti.nn', 'O': 'O.nn'}

        input_str = potential.predict_input_string(
            xsf_files=['structure.xsf'],
            eval_forces=True,
            config=PredictionConfig()
        )

        self.assertIn('FORCES', input_str)

    def test_predict_input_string_with_options(self):
        """Test predict_input_string with optional parameters."""
        from aenet.mlip import PredictionConfig

        potential = ANNPotential(self.simple_arch)
        potential._potential_paths = {'Ti': 'Ti.nn', 'O': 'O.nn'}

        config = PredictionConfig(
            potential_format='ascii',
            timing=True,
            print_atomic_energies=True,
            debug=True,
            verbosity=2
        )

        input_str = potential.predict_input_string(
            xsf_files=['structure.xsf'],
            eval_forces=False,
            config=config
        )

        self.assertIn('format=ascii', input_str)
        self.assertIn('TIMING', input_str)
        self.assertIn('PRINT_ATOMIC_ENERGIES', input_str)
        self.assertIn('DEBUG', input_str)
        self.assertIn('VERBOSITY 2', input_str)

    def test_write_predict_input_file(self):
        """Test writing predict.in file to disk."""
        from aenet.mlip import PredictionConfig

        potential = ANNPotential(self.simple_arch)
        potential._potential_paths = {'Ti': 'Ti.nn', 'O': 'O.nn'}

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PredictionConfig(verbosity=1)
            potential.write_predict_input_file(
                xsf_files=['structure.xsf'],
                eval_forces=True,
                config=config,
                workdir=tmpdir,
                filename='predict.in'
            )

            output_file = os.path.join(tmpdir, 'predict.in')
            self.assertTrue(os.path.exists(output_file))

            # Read and verify content
            with open(output_file, 'r') as f:
                content = f.read()
            self.assertIn('TYPES', content)
            self.assertIn('FORCES', content)
            self.assertIn('NETWORKS', content)

    @patch('os.path.exists')
    def test_predict_missing_potentials_error(self, mock_exists):
        """Test predict raises error when no potentials available."""
        mock_exists.return_value = True

        potential = ANNPotential(self.simple_arch)
        # Don't set _potential_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with self.assertRaises(ValueError) as context:
                    potential.predict(['structure.xsf'])
            finally:
                os.chdir(cwd)

        self.assertIn('No potential paths available', str(context.exception))

    @patch('aenet.mlip.cfg')
    @patch('os.path.exists')
    def test_predict_missing_executable_error(self, mock_exists, mock_cfg):
        """Test predict raises error when predict.x not found."""
        mock_exists.return_value = False
        mock_cfg.read.return_value = {'predict_x_path': '/path/to/predict.x'}

        potential = ANNPotential(self.simple_arch)
        potential._potential_paths = {'Ti': 'Ti.nn', 'O': 'O.nn'}

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with self.assertRaises(FileNotFoundError) as context:
                    potential.predict(['structure.xsf'])
            finally:
                os.chdir(cwd)

        self.assertIn('Cannot find `predict.x`', str(context.exception))

    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(__file__),
                                    'data', 'Ti.nn.ascii')),
        "Test data not available"
    )
    def test_predict_output_parsing(self):
        """Test PredictOut parsing with real output file."""

        # This test uses real test data to verify output parsing
        test_dir = os.path.dirname(__file__)
        data_dir = os.path.join(test_dir, 'data')

        # Check if we have required test files
        ti_potential = os.path.join(data_dir, 'Ti.nn.ascii')
        o_potential = os.path.join(data_dir, 'O.nn.ascii')
        xsf_dir = os.path.join(data_dir, 'xsf-TiO2')

        self.assertTrue(os.path.exists(ti_potential))
        self.assertTrue(os.path.exists(o_potential))
        self.assertTrue(os.path.exists(xsf_dir))


class TestPredictIntegration(unittest.TestCase):
    """Integration tests for full prediction workflow."""

    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(__file__),
                                    'data', 'predict.out')),
        "Test data not available"
    )
    def test_predict_integration(self):
        """Integration test: PredictOut parsing with real predict.out file."""
        from aenet.io.predict import PredictOut

        # Use real predict.out file from test data
        test_dir = os.path.dirname(__file__)
        data_dir = os.path.join(test_dir, 'data')
        predict_out_file = os.path.join(data_dir, 'predict.out')

        # Get paths to XSF files referenced in predict.out
        xsf_dir = os.path.join(data_dir, 'xsf-TiO2')
        xsf_files = [
            os.path.join(xsf_dir, 'structure-001.xsf'),
            os.path.join(xsf_dir, 'structure-002.xsf'),
            os.path.join(xsf_dir, 'structure-003.xsf')
        ]

        # Parse the prediction output
        results = PredictOut.from_file(
            predict_out_file, structure_paths=xsf_files)

        # Verify results
        self.assertEqual(results.num_structures, 3)
        self.assertEqual(len(results.cohesive_energy), 3)
        self.assertEqual(len(results.total_energy), 3)
        self.assertEqual(len(results.forces), 3)

        # Check that energies match expected values from the file
        # Structure 1: 23 atoms
        self.assertEqual(results.num_atoms(0), 23)
        self.assertAlmostEqual(
            results.cohesive_energy[0], -192.78242329, places=5)
        self.assertAlmostEqual(
            results.total_energy[0], -19517.16578343, places=5)

        # Structure 2: 46 atoms
        self.assertEqual(results.num_atoms(1), 46)
        self.assertAlmostEqual(
            results.cohesive_energy[1], -389.72446909, places=5)
        self.assertAlmostEqual(
            results.total_energy[1], -39038.49118938, places=5)

        # Structure 3: 23 atoms
        self.assertEqual(results.num_atoms(2), 23)
        self.assertAlmostEqual(
            results.cohesive_energy[2], -193.42936699, places=5)
        self.assertAlmostEqual(
            results.total_energy[2], -19517.81272714, places=5)

        # Check that energies are reasonable (not NaN or inf)
        for energy in results.cohesive_energy:
            self.assertFalse(np.isnan(energy))
            self.assertFalse(np.isinf(energy))

        # Check coordinates and atom types are present
        for i in range(3):
            self.assertGreater(len(results.coords[i]), 0)
            self.assertGreater(len(results.atom_types[i]), 0)
            self.assertEqual(len(results.coords[i]), results.num_atoms(i))
            self.assertEqual(len(results.atom_types[i]), results.num_atoms(i))


if __name__ == '__main__':
    unittest.main()
