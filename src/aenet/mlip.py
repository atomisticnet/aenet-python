"""
Train and use machine-learning interatomic potentials.

"""

import os
from typing import Dict, List, Tuple, Literal, Any, Optional, Union
from dataclasses import dataclass
import subprocess
import tempfile
import glob
import shutil
import time
from tqdm import tqdm

from . import config as cfg
from .trainset import TrnSet
from .util import cd

__author__ = "The aenet developers"
__email__ = "aenet@atomistic.net"
__date__ = "2025-10-31"
__version__ = "0.2"


Activation = Literal['tanh', 'linear', 'relu', 'gelu', 'twist']
ANNArchitecture = Dict[str, List[Tuple[int, Activation]]]


@dataclass
class PredictionConfig:
    """
    Configuration for ANN potential predictions.

    Centralizes all prediction parameters with built-in validation.

    Parameters
    ----------
    potential_paths : Dict[str, str], optional
        Mapping of element symbols to potential file paths.
        If None, uses paths stored in ANNPotential after training.
        Default: None
    potential_format : str, optional
        Format of potential files: 'ascii' or None (binary).
        If 'ascii', predict.x will convert to binary automatically.
        Default: None (binary)
    timing : bool, optional
        Enable detailed timing output. Default: False
    print_atomic_energies : bool, optional
        Print per-atom energy contributions. Default: False
    debug : bool, optional
        Enable debug output. Default: False
    verbosity : int, optional
        Verbosity level: 0 (low), 1 (normal), or 2 (high).
        Default: 1
    batch_size : int, optional
        Batch size for PyTorch predictions (PyTorch API only).
        Default: 32

    PyTorch-only DataLoader knobs (ignored by Fortran API)
    ------------------------------------------------------

    num_workers : int, optional
        Number of DataLoader workers. Default: 0
        The default of 0 corresponds to single-threaded loading, and
        any value >0 results in that many worker processes. Each worker
        independently featurizes structures in parallel (neighbor lists,
        Chebyshev descriptors, etc.), then feeds batches to the main
        process for GPU/model forward.  Often
        `num_workers = min(4, cpu_cores//2)` is a good starting point for
        CPU systems.  More workers require more memory.
    prefetch_factor : int, optional
        Number of batches to prefetch per worker. Default: 2
        This controls how many batches each worker loads in advance.
        Higher values can improve throughput for complex featurizations
        or slower storage, at the cost of higher memory usage.  The default
        value should be suitable for most use cases.  This parameter is
        only used if `num_workers > 0`.
    persistent_workers : bool, optional
        Whether to keep DataLoader workers alive between epochs.
        Default: True
        This can improve performance when making multiple prediction
        calls in succession, at the cost of slightly higher memory usage.
        This parameter is only used if `num_workers > 0`.

    Raises
    ------
    ValueError
        If parameters are out of valid ranges.
    """
    potential_paths: Optional[Dict[str, str]] = None
    potential_format: Optional[str] = None
    timing: bool = False
    print_atomic_energies: bool = False
    debug: bool = False
    verbosity: int = 1
    batch_size: int = 32
    # PyTorch-only DataLoader knobs (ignored by Fortran API)
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = True

    def __post_init__(self):
        """Validate parameters after initialization."""
        # Validate verbosity
        if self.verbosity not in [0, 1, 2]:
            raise ValueError(
                f"verbosity must be 0, 1, or 2, got {self.verbosity}"
            )

        # Validate potential_format
        if self.potential_format is not None:
            valid_formats = ['ascii', 'ASCII']
            if self.potential_format not in valid_formats:
                raise ValueError(
                    f"potential_format must be 'ascii', 'ASCII', or None, "
                    f"got '{self.potential_format}'"
                )

        # Validate batch_size
        if self.batch_size < 1:
            raise ValueError(
                f"batch_size must be >= 1, got {self.batch_size}"
            )
        # Validate PyTorch DataLoader knobs (ignored by Fortran backend)
        if self.num_workers < 0:
            raise ValueError(
                f"num_workers must be >= 0, got {self.num_workers}"
            )
        if self.prefetch_factor < 1:
            raise ValueError(
                f"prefetch_factor must be >= 1, got {self.prefetch_factor}"
            )

    def user_changed(self) -> Dict[str, Any]:
        """
        Return configuration parameters that differ from their default values.

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping parameter names to their current values,
            only including parameters that differ from defaults.

        Examples
        --------
        >>> config = PredictionConfig(timing=True, batch_size=64)
        >>> config.user_changed()
        {'timing': True, 'batch_size': 64}
        """
        defaults = PredictionConfig()
        changed = {}
        for field_name in [
            'potential_paths', 'potential_format', 'timing',
            'print_atomic_energies', 'debug', 'verbosity', 'batch_size',
            'num_workers', 'prefetch_factor', 'persistent_workers'
        ]:
            current_val = getattr(self, field_name)
            default_val = getattr(defaults, field_name)
            if current_val != default_val:
                changed[field_name] = current_val
        return changed


@dataclass
class TrainingConfig:
    """
    Configuration for ANN potential training.

    Centralizes all training parameters with built-in validation.

    Parameters
    ----------
    iterations : int, optional
        Maximum number of training iterations. Default: 0
    method : TrainingMethod, optional
        Training method configuration (Adam, BFGS, EKF, LM, or OnlineSD).
        Default: None (will use Adam with defaults)
    testpercent : int, optional
        Percentage of data for test set (0-100). Default: 0
    max_energy : float, optional
        Exclude structures with energy above this threshold. Default: None
    sampling : str, optional
        Sampling method for training set. Options: 'sequential', 'random',
        'weighted', 'energy'. Default: None (aenet default: sequential)
    timing : bool, optional
        Enable detailed timing output. Default: False
    save_energies : bool, optional
        Save predicted energies for training/test sets. Default: False

    Raises
    ------
    ValueError
        If parameters are out of valid ranges.
    """
    iterations: int = 0
    method: Optional['TrainingMethod'] = None
    testpercent: int = 0
    max_energy: Optional[float] = None
    sampling: Optional[str] = None
    timing: bool = False
    save_energies: bool = False

    def __post_init__(self):
        """Validate parameters after initialization."""
        # Validate testpercent
        if not 0 <= self.testpercent <= 100:
            raise ValueError(
                f"testpercent must be 0-100, got {self.testpercent}"
            )

        # Validate sampling
        if self.sampling is not None:
            valid_sampling = ['sequential', 'random', 'weighted', 'energy']
            if self.sampling not in valid_sampling:
                raise ValueError(
                    f"sampling must be one of {valid_sampling}, "
                    f"got '{self.sampling}'"
                )

        # Validate iterations
        if self.iterations < 0:
            raise ValueError(
                f"iterations must be >= 0, got {self.iterations}"
            )

        # Default method to Adam if not provided
        if self.method is None:
            self.method = Adam()


@dataclass
class TrainingMethod:
    """
    Base class for training method configurations.

    Each training method subclass encodes both the method name and its
    parameters with appropriate defaults.
    """

    @property
    def method_name(self) -> str:
        """Return the method string used by aenet's train.x."""
        raise NotImplementedError

    def to_params_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to parameter dictionary.

        Returns
        -------
        dict
            Dictionary of parameter names and values for the METHOD line.
        """
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


@dataclass
class OnlineSD(TrainingMethod):
    """
    Online steepest descent optimizer.

    Parameters
    ----------
    gamma : float, optional
        Learning rate. Default: 1.0e-5
    alpha : float, optional
        Momentum parameter. Default: 0.25
    """
    gamma: float = 1.0e-5
    alpha: float = 0.25

    @property
    def method_name(self) -> str:
        return "online_sd"


@dataclass
class Adam(TrainingMethod):
    """
    ADAM optimizer.

    Parameters
    ----------
    mu : float, optional
        Learning rate. Default: 0.001
    b1 : float, optional
        Exponential decay rate for first moment estimates. Default: 0.9
    b2 : float, optional
        Exponential decay rate for second moment estimates. Default: 0.999
    eps : float, optional
        Small constant for numerical stability. Default: 1.0e-8
    batchsize : int, optional
        Number of structures per batch. Default: 16
    samplesize : int, optional
        Number of structures to sample per epoch. Default: 100
    """
    mu: float = 0.001
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1.0e-8
    batchsize: int = 16
    samplesize: int = 100

    @property
    def method_name(self) -> str:
        return "adam"


@dataclass
class EKF(TrainingMethod):
    """
    Extended Kalman filter optimizer.

    Parameters
    ----------
    lambda_ : float, optional
        Forgetting factor. Default: 0.99
    lambda0 : float, optional
        Initial forgetting factor. Default: 0.999
    P : float, optional
        Initial covariance. Default: 100.0
    mnoise : float, optional
        Measurement noise. Default: 0.0
    pnoise : float, optional
        Process noise. Default: 1.0e-5
    wgmax : int, optional
        Maximum weight change. Default: 500
    """
    lambda_: float = 0.99
    lambda0: float = 0.999
    P: float = 100.0
    mnoise: float = 0.0
    pnoise: float = 1.0e-5
    wgmax: int = 500

    @property
    def method_name(self) -> str:
        return "ekf"

    def to_params_dict(self) -> Dict[str, Any]:
        """Convert to params dict, handling lambda_ -> lambda."""
        params = super().to_params_dict()
        if 'lambda_' in params:
            params['lambda'] = params.pop('lambda_')
        return params


@dataclass
class LM(TrainingMethod):
    """
    Levenberg-Marquardt optimizer.

    Parameters
    ----------
    batchsize : int, optional
        Number of structures per batch. Default: 256
    learnrate : float, optional
        Learning rate. Default: 0.1
    iter : int, optional
        Number of iterations per epoch. Default: 3
    conv : float, optional
        Convergence criterion. Default: 1e-3
    adjust : int, optional
        Adjustment parameter. Default: 5
    """
    batchsize: int = 256
    learnrate: float = 0.1
    iter: int = 3
    conv: float = 1e-3
    adjust: int = 5

    @property
    def method_name(self) -> str:
        return "lm"


@dataclass
class BFGS(TrainingMethod):
    """
    L-BFGS-B optimizer.

    This method has no configurable parameters.

    Note
    ----
    Not supported on ARM-based Macs.
    """

    @property
    def method_name(self) -> str:
        return "bfgs"

    def to_params_dict(self) -> Dict[str, Any]:
        """BFGS has no parameters."""
        return {}


class ANNPotential(object):
    """
    Artificial Neural Network Potential for machine-learning interatomic
    potentials.

    This class facilitates the training of aenet ANN potentials by managing
    network architectures, generating input files, and orchestrating the
    training process via aenet's train.x executable.

    Parameters
    ----------
    arch : ANNArchitecture
        Dictionary mapping atomic species (element symbols) to their network
        architectures. Each architecture is a list of tuples (num_nodes,
        activation) defining the hidden layers. The final output layer
        (linear, 1 node) is automatically added by aenet and should not
        be included.

    Attributes
    ----------
    arch : ANNArchitecture
        The network architecture specification.
    num_types : int
        Number of atomic species in the potential.
    atom_types : List[str]
        List of atomic species symbols.

    Examples
    --------
    >>> arch = {
    ...     'Si': [(10, 'tanh'), (10, 'tanh')],
    ...     'O': [(10, 'tanh'), (10, 'tanh')]
    ... }
    >>> potential = ANNPotential(arch)
    >>> potential.train(trnset_file='data.train', iterations=1000)
    """

    # Supported PredictionConfig options for Fortran API
    _supported_config_options = {
        'potential_paths', 'potential_format', 'timing',
        'print_atomic_energies', 'debug', 'verbosity'
    }

    def __init__(self, arch: ANNArchitecture):
        self.arch = arch
        self._potential_paths = None
        self._potential_format = None
        self._from_files = False

    @property
    def num_types(self) -> int:
        """Number of atomic types in the potential."""
        return len(self.arch)

    @property
    def atom_types(self) -> List[str]:
        """List of atomic types in the potential."""
        return list(self.arch.keys())

    def train(self,
              trnset_file: os.PathLike = 'data.train',
              config: Optional[TrainingConfig] = None,
              workdir: os.PathLike = None,
              output_file: os.PathLike = 'train.out',
              num_processes: Optional[int] = None):
        """
        Train the ANN potential using aenet's train.x executable.

        This method orchestrates the complete training workflow:
        validates inputs, generates the train.in configuration file,
        executes train.x, monitors progress with a progress bar, and
        collects output files (.nn potentials, energies, timing data)
        into the current directory.

        Parameters
        ----------
        trnset_file : os.PathLike, optional
            Path to the training set file (ASCII or HDF5 format).
            Default: 'data.train'
        config : TrainingConfig, optional
            Training configuration object containing all training parameters.
            Default: None (will use TrainingConfig with defaults)
        workdir : os.PathLike, optional
            Directory for temporary training files. If None, a temporary
            directory is created and removed after training. If specified,
            the directory is kept. Default: None
        output_file : os.PathLike, optional
            File path to save train.x standard output. Default: 'train.out'
        num_processes : int, optional
            Number of MPI processes to use for parallel training. Requires
            MPI support to be enabled in the configuration (mpi_enabled=True).
            If None, training runs without MPI. Default: None

        Raises
        ------
        FileNotFoundError
            If trnset_file doesn't exist or train.x executable is not
            configured.
        ValueError
            If training set contains species not in the architecture, or
            if config parameters are invalid.

        Notes
        -----
        - Requires configured aenet installation. Use 'aenet config' to
          set paths.
        - Output files (.nn, energies.*, train.time) are moved to current
          directory.
        - Training progress is displayed with a progress bar.

        Examples
        --------
        >>> from aenet.mlip import ANNPotential, TrainingConfig, Adam, BFGS
        >>> potential = ANNPotential({'Si': [(10, 'tanh'), (10, 'tanh')]})
        >>> # Use defaults
        >>> potential.train('data.train')
        >>> # Use TrainingConfig
        >>> config = TrainingConfig(iterations=1000, method=Adam(mu=0.005))
        >>> potential.train('data.train', config=config)
        >>> # Or inline
        >>> potential.train('data.train',
        ...                 config=TrainingConfig(iterations=1000,
        ...                                       sampling='random'))
        """

        # Use default config if not provided
        if config is None:
            config = TrainingConfig()

        # Validate num_processes
        if num_processes is not None and num_processes <= 0:
            raise ValueError(
                f"num_processes must be > 0, got {num_processes}")

        if workdir is None:
            workdir = tempfile.mkdtemp(dir='.')
            rm_tmp_files = True
        else:
            if not os.path.exists(workdir):
                os.makedirs(workdir, exist_ok=True)
            rm_tmp_files = False

        if not os.path.exists(trnset_file):
            if os.path.exists(os.path.join(workdir, trnset_file)):
                trnset_file = os.path.join(workdir, trnset_file)
            else:
                raise FileNotFoundError(
                    'Training set file not found: {}'.format(trnset_file))

        # check if all species in the training set are in the architecture
        with TrnSet.from_file(trnset_file) as ts:
            if not set(ts.atom_types) <= set(self.atom_types):
                raise ValueError(
                    'Not all species in the training set are in the '
                    'MLIP architecture. '
                    'Training set: {} '.format(ts.atom_types),
                    'Architecture: {}'.format(self.atom_types))

        aenet_paths = cfg.read('aenet')
        if not os.path.exists(aenet_paths['train_x_path']):
            raise FileNotFoundError(
                "Cannot find `train.x`. Configure with `aenet config`.")

        self.write_train_input_file(trnset_file=trnset_file,
                                    config=config,
                                    workdir=workdir)

        # Construct command for MPI or non-MPI execution
        train_x_path = aenet_paths['train_x_path']
        if num_processes and aenet_paths.get('mpi_enabled', False):
            launcher = aenet_paths['mpi_launcher']
            command = launcher.format(
                num_proc=num_processes,
                exec=train_x_path
            ).split() + ['train.in']
        else:
            command = [train_x_path, 'train.in']

        # Call `train.x` and perform the training
        with cd(workdir) as cm:
            outfile = os.path.join(cm['origin'], output_file)
            errfile = 'errors.out'
            with open(outfile, 'w') as out, open(errfile, 'w') as err:
                proc = subprocess.Popen(command, stdout=out, stderr=err)

        def get_progress_percentage():
            epoch = len(glob.glob(os.path.join(
                workdir, '{}.nn-*'.format(self.atom_types[0]))))
            return 100*(epoch/config.iterations)

        progress_bar = tqdm(total=100, desc="Training", ncols=80)
        while proc.poll() is None:
            percent = get_progress_percentage()
            progress_bar.n = percent
            progress_bar.refresh()
            time.sleep(10)

        progress_bar.n = 100
        progress_bar.refresh()
        progress_bar.close()

        for f in glob.glob(os.path.join(workdir, '*.nn')):
            shutil.move(f, os.curdir)

        for f in glob.glob(os.path.join(workdir, 'energies.*')):
            shutil.move(f, os.curdir)

        for f in glob.glob(os.path.join(workdir, 'train.time')):
            shutil.move(f, os.curdir)

        # Store potential paths after successful training
        self._potential_paths = {el: f'{el}.nn' for el in self.atom_types}

        if rm_tmp_files:
            shutil.rmtree(workdir)

    def train_input_string(self,
                           trnset_file: os.PathLike = 'data.train',
                           config: Optional[TrainingConfig] = None,
                           workdir: os.PathLike = '.'):
        """
        Generate the content of a train.in input file for aenet's train.x.

        Creates a properly formatted input file string containing training
        parameters, network architectures, and dataset specifications.

        Parameters
        ----------
        trnset_file : os.PathLike, optional
            Path to the training set file. Default: 'data.train'
        config : TrainingConfig, optional
            Training configuration object containing all training parameters.
            Default: None (will use TrainingConfig with defaults)
        workdir : os.PathLike, optional
            Working directory (used to make trnset_file path relative).
            Default: '.'

        Returns
        -------
        str
            Formatted train.in file content.

        Examples
        --------
        >>> from aenet.mlip import ANNPotential, TrainingConfig, Adam
        >>> potential = ANNPotential({'Si': [(10, 'tanh')]})
        >>> config = TrainingConfig(iterations=1000, method=Adam(mu=0.005))
        >>> input_str = potential.train_input_string(
        ...     trnset_file='data.train',
        ...     config=config
        ... )
        """

        # Use default config if not provided
        if config is None:
            config = TrainingConfig()

        trnset_file = os.path.relpath(trnset_file, workdir)
        train_in = "TRAININGSET \"{}\"\n".format(trnset_file)
        train_in += "TESTPERCENT {}\n".format(config.testpercent)
        train_in += "ITERATIONS {}\n\n".format(config.iterations)
        if config.max_energy is not None:
            train_in += "MAXENERGY {}\n\n".format(config.max_energy)
        if config.sampling is not None:
            train_in += "SAMPLING {}\n\n".format(config.sampling)
        if config.timing:
            train_in += "TIMING\n\n"
        if config.save_energies:
            train_in += "SAVE_ENERGIES\n\n"

        # Generate METHOD line from TrainingMethod object
        method_name = config.method.method_name
        method_params = config.method.to_params_dict()
        if method_params:
            paramstring = " ".join(["{}={}".format(k, v)
                                    for k, v in method_params.items()])
            train_in += "METHOD\n{} {}\n\n".format(method_name, paramstring)
        else:
            train_in += "METHOD\n{}\n\n".format(method_name)

        train_in += "NETWORKS\n"
        for el in self.arch:
            line = "{:2s}  {:5s}  {:4d}   ".format(
                el, "{}.nn".format(el), len(self.arch[el]))
            for layer in self.arch[el]:
                line += " {}:{}".format(*layer)
            train_in += line + "\n"
        train_in += "\n"
        return train_in

    def write_train_input_file(self,
                               trnset_file: os.PathLike = 'data.train',
                               config: Optional[TrainingConfig] = None,
                               workdir: os.PathLike = '.',
                               filename: os.PathLike = 'train.in'):
        """
        Write a train.in input file for aenet's train.x executable.

        This method generates the input file content using train_input_string()
        and writes it to the specified location.

        Parameters
        ----------
        trnset_file : os.PathLike, optional
            Path to the training set file. Default: 'data.train'
        config : TrainingConfig, optional
            Training configuration object containing all training parameters.
            Default: None (will use TrainingConfig with defaults)
        workdir : os.PathLike, optional
            Directory where the train.in file will be written. Created if
            it doesn't exist. Default: '.'
        filename : os.PathLike, optional
            Name of the output file. Default: 'train.in'

        Raises
        ------
        AssertionError
            If the output file already exists.

        See Also
        --------
        train_input_string : Generate train.in content as a string.
        """

        if not os.path.exists(workdir):
            os.makedirs(workdir, exist_ok=True)

        outfile = os.path.join(workdir, filename)
        if os.path.exists(outfile):
            raise AssertionError('File already exists: {}'.format(outfile))

        train_in = self.train_input_string(trnset_file=trnset_file,
                                           config=config,
                                           workdir=workdir)

        with open(outfile, 'w') as fp:
            fp.write(train_in)

    @classmethod
    def from_files(cls, potential_paths: Dict[str, str],
                   potential_format: str = None) -> 'ANNPotential':
        """
        Create an ANNPotential from existing potential files.

        This factory method creates an instance for prediction-only use cases
        where you have pre-trained potential files but may not have the
        architecture information.

        Parameters
        ----------
        potential_paths : Dict[str, str]
            Dictionary mapping element symbols to potential file paths.
        potential_format : str, optional
            Format of the potential files: 'ascii' (or 'ASCII') for ASCII
            format, or None for binary format. This format will be used
            as the default for predictions unless overridden in
            PredictionConfig. Default: None (binary)

        Returns
        -------
        ANNPotential
            Instance configured for predictions with the given potentials.

        Raises
        ------
        FileNotFoundError
            If any of the potential files do not exist.

        Notes
        -----
        Architecture information cannot be extracted from binary files,
        so this instance has limited functionality (predict only, cannot
        re-train).

        Examples
        --------
        >>> # Load ASCII format potentials
        >>> potential = ANNPotential.from_files({
        ...     'Ti': 'Ti.nn.ascii',
        ...     'O': 'O.nn.ascii'
        ... }, potential_format='ascii')
        >>> # Format is remembered for predictions
        >>> results = potential.predict(structures, eval_forces=True)

        >>> # Load binary format potentials
        >>> potential = ANNPotential.from_files({
        ...     'Ti': 'Ti.nn',
        ...     'O': 'O.nn'
        ... })  # potential_format=None for binary
        """
        # Validate that all potential files exist
        missing_files = []
        for element, path in potential_paths.items():
            if not os.path.exists(path):
                missing_files.append(f"{element}: {path}")

        if missing_files:
            raise FileNotFoundError(
                "Potential file(s) not found:\n" +
                "\n".join(f"  - {f}" for f in missing_files)
            )

        # Create with minimal/empty architecture
        instance = cls(arch={el: [] for el in potential_paths.keys()})
        instance._potential_paths = potential_paths
        instance._potential_format = potential_format
        instance._from_files = True
        return instance

    def predict(self,
                structures: Union[List, List[os.PathLike]],
                eval_forces: bool = False,
                config: Optional[PredictionConfig] = None,
                workdir: os.PathLike = None,
                output_file: os.PathLike = 'predict.out',
                num_processes: Optional[int] = None):
        """
        Predict energies (and optionally forces) for structures using the
        trained ANN potential.

        This method orchestrates the prediction workflow: accepts either
        AtomicStructure objects or XSF file paths, generates temporary XSF
        files if needed, creates the predict.in input file, executes
        predict.x, and parses the results.

        Parameters
        ----------
        structures : List[AtomicStructure] or List[os.PathLike]
            Either a list of AtomicStructure objects or a list of paths
            to XSF structure files.
        eval_forces : bool, optional
            If True, compute and return atomic forces. Default: False
        config : PredictionConfig, optional
            Prediction configuration object. If None, uses defaults.
            Default: None
        workdir : os.PathLike, optional
            Directory for prediction files. If None, a temporary directory
            is created and removed after prediction. Default: None
        output_file : os.PathLike, optional
            File path to save predict.x standard output.
            Default: 'predict.out'
        num_processes : int, optional
            Number of MPI processes for parallel predictions. Requires
            MPI support enabled in configuration. Default: None

        Returns
        -------
        PredictOut
            Parsed prediction results containing energies, forces, and
            structure information.

        Raises
        ------
        FileNotFoundError
            If predict.x executable is not configured or structure files
            don't exist.
        ValueError
            If no potential paths are available or configuration is invalid.

        Examples
        --------
        >>> # After training
        >>> potential = ANNPotential(arch)
        >>> potential.train(trnset_file='data.train')
        >>> results = potential.predict(structures, eval_forces=True)
        >>> print(results.cohesive_energy)

        >>> # Or load existing potentials
        >>> potential = ANNPotential.from_files({'Ti': 'Ti.nn', 'O': 'O.nn'})
        >>> results = potential.predict(['structure.xsf'], eval_forces=True)
        >>> enriched_strucs = results.to_structures()
        """
        from .io.predict import PredictOut
        from .geometry import AtomicStructure
        from .formats.xsf import XSFParser

        # Use default config if not provided
        if config is None:
            config = PredictionConfig()

        # Warn about unsupported config options
        changed_options = config.user_changed()
        unsupported = (set(changed_options.keys())
                       - self._supported_config_options)
        if unsupported:
            import warnings
            unsupported_list = ', '.join(sorted(unsupported))
            warnings.warn(
                f"The following PredictionConfig parameters are not "
                f"supported by the Fortran API and will be ignored: "
                f"{unsupported_list}",
                UserWarning
            )

        # Set potential paths if not provided in config
        if config.potential_paths is None:
            if self._potential_paths is None:
                raise ValueError(
                    "No potential paths available. Either train first or "
                    "provide paths in PredictionConfig.")
            config.potential_paths = self._potential_paths

        # Use stored format if not specified in config
        if (config.potential_format is None
                and self._potential_format is not None):
            config = PredictionConfig(
                potential_paths=config.potential_paths,
                potential_format=self._potential_format,
                timing=config.timing,
                print_atomic_energies=config.print_atomic_energies,
                debug=config.debug,
                verbosity=config.verbosity
            )

        # Validate num_processes
        if num_processes is not None and num_processes <= 0:
            raise ValueError(
                f"num_processes must be > 0, got {num_processes}")

        # Check predict.x exists
        aenet_paths = cfg.read('aenet')
        if not os.path.exists(aenet_paths['predict_x_path']):
            raise FileNotFoundError(
                "Cannot find `predict.x`. Configure with `aenet config`.")

        # Setup workdir
        if workdir is None:
            workdir = tempfile.mkdtemp(dir='.')
            rm_tmp_files = True
        else:
            if not os.path.exists(workdir):
                os.makedirs(workdir, exist_ok=True)
            rm_tmp_files = False

        # Handle structure input: convert AtomicStructure objects to XSF files
        xsf_files = []
        structures_are_objects = False
        if structures and isinstance(structures[0], AtomicStructure):
            structures_are_objects = True
            xsf_parser = XSFParser()
            for i, struc in enumerate(structures):
                xsf_file = os.path.join(workdir, f'structure{i:04d}.xsf')
                xsf_parser.write(struc, outfile=xsf_file)
                xsf_files.append(os.path.relpath(
                    os.path.abspath(xsf_file), os.path.abspath(workdir)))
        else:
            # Structures are file paths - validate and make relative to workdir
            for f in structures:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Structure file not found: {f}")
                xsf_files.append(os.path.relpath(
                    os.path.abspath(f), os.path.abspath(workdir)))

        # Generate predict.in file
        self.write_predict_input_file(
            xsf_files=xsf_files,
            eval_forces=eval_forces,
            config=config,
            workdir=workdir)

        # Construct command for MPI or non-MPI execution
        predict_x_path = aenet_paths['predict_x_path']
        if num_processes and aenet_paths.get('mpi_enabled', False):
            launcher = aenet_paths['mpi_launcher']
            command = launcher.format(
                num_proc=num_processes,
                exec=predict_x_path
            ).split() + ['predict.in']
        else:
            command = [predict_x_path, 'predict.in']

        # Execute predict.x
        with cd(workdir) as cm:
            outfile = os.path.join(cm['origin'], output_file)
            errfile = 'errors.out'
            with open(outfile, 'w') as out, open(errfile, 'w') as err:
                subprocess.run(command, stdout=out, stderr=err)

        # Parse results - need absolute paths for structure reconstruction
        if structures_are_objects:
            # Use the temporary XSF files we created
            structure_paths = [os.path.abspath(os.path.join(workdir, f))
                               for f in xsf_files]
        else:
            # Use the original paths provided by user
            structure_paths = [os.path.abspath(f) for f in structures]

        results = PredictOut.from_file(
            outfile,
            structure_paths=structure_paths)

        if rm_tmp_files:
            shutil.rmtree(workdir)

        return results

    def predict_input_string(self,
                             xsf_files: List[os.PathLike],
                             eval_forces: bool = False,
                             config: Optional[PredictionConfig] = None,
                             workdir: os.PathLike = '.') -> str:
        """
        Generate the content of a predict.in input file for aenet's predict.x.

        Parameters
        ----------
        xsf_files : List[os.PathLike]
            List of XSF structure file paths (relative to workdir).
        eval_forces : bool, optional
            If True, enable force evaluation. Default: False
        config : PredictionConfig, optional
            Prediction configuration. Default: None (uses defaults)
        workdir : os.PathLike, optional
            Working directory for relative paths. Default: '.'

        Returns
        -------
        str
            Formatted predict.in file content.
        """
        if config is None:
            config = PredictionConfig()

        if config.potential_paths is None:
            if self._potential_paths is None:
                raise ValueError("No potential paths available.")
            config.potential_paths = self._potential_paths

        # Get atom types from potential paths
        atom_types = list(config.potential_paths.keys())

        # Build input file string
        predict_in = "TYPES\n{}\n".format(len(atom_types))
        for typ in atom_types:
            predict_in += "{}\n".format(typ)

        predict_in += "\nNETWORKS"
        if config.potential_format is not None:
            predict_in += " format={}".format(config.potential_format)
        predict_in += "\n"
        for typ in atom_types:
            rel_path = os.path.relpath(config.potential_paths[typ], workdir)
            predict_in += "{} \"{}\"\n".format(typ, rel_path)

        if eval_forces:
            predict_in += "\nFORCES\n"

        if config.timing:
            predict_in += "\nTIMING\n"

        if config.print_atomic_energies:
            predict_in += "\nPRINT_ATOMIC_ENERGIES\n"

        if config.debug:
            predict_in += "\nDEBUG\n"

        predict_in += "\nVERBOSITY {}\n".format(config.verbosity)

        predict_in += "\nFILES\n{}\n".format(len(xsf_files))
        for xsf_file in xsf_files:
            predict_in += "{}\n".format(xsf_file)

        return predict_in

    def write_predict_input_file(self,
                                 xsf_files: List[os.PathLike],
                                 eval_forces: bool = False,
                                 config: Optional[PredictionConfig] = None,
                                 workdir: os.PathLike = '.',
                                 filename: os.PathLike = 'predict.in'):
        """
        Write a predict.in input file for aenet's predict.x executable.

        Parameters
        ----------
        xsf_files : List[os.PathLike]
            List of XSF structure file paths (relative to workdir).
        eval_forces : bool, optional
            If True, enable force evaluation. Default: False
        config : PredictionConfig, optional
            Prediction configuration. Default: None (uses defaults)
        workdir : os.PathLike, optional
            Directory where the predict.in file will be written.
            Default: '.'
        filename : os.PathLike, optional
            Name of the output file. Default: 'predict.in'

        Raises
        ------
        AssertionError
            If the output file already exists.
        """
        if not os.path.exists(workdir):
            os.makedirs(workdir, exist_ok=True)

        outfile = os.path.join(workdir, filename)
        if os.path.exists(outfile):
            raise AssertionError('File already exists: {}'.format(outfile))

        predict_in = self.predict_input_string(
            xsf_files=xsf_files,
            eval_forces=eval_forces,
            config=config,
            workdir=workdir)

        with open(outfile, 'w') as fp:
            fp.write(predict_in)
