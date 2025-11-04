"""
Handling input and output files for/from aenet's 'train.x' tool.

This module provides classes for parsing and analyzing training output from
both the Fortran-based train.x and PyTorch-based training workflows.
"""

import os
import re
from typing import Dict, List, Optional, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .. import config as cfg

__author__ = "Nongnuch Artrith, Alexander Urban"
__email__ = "aenet@atomistic.net"
__date__ = "2021-02-17"
__version__ = "0.2"


class TrainOutput(object):
    """
    Deprecated class name. Use 'TrainOut' instead.

    This class is maintained for backward compatibility but will be removed
    in a future version. New code should use TrainOut directly.

    Parameters
    ----------
    *args : tuple
        Positional arguments passed to TrainOut
    **kwargs : dict
        Keyword arguments passed to TrainOut

    See Also
    --------
    TrainOut : Replacement class with full documentation
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        import warnings
        warnings.warn(
            "The 'TrainOutput' class is deprecated. Use 'TrainOut' instead.",
            DeprecationWarning)
        self._train_out = TrainOut(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._train_out, name)


class TrainOut(object):
    """
    Parser and representation of training output from aenet or PyTorch.

    This class handles both file-based output (from Fortran train.x) and
    in-memory data (from PyTorch training), providing a unified interface
    for analyzing training results.

    Parameters
    ----------
    path : os.PathLike, optional
        Path to train.x output file (for Fortran backend). If provided,
        reads training errors from this file. Default: None
    energies_train_files : List[os.PathLike], optional
        Paths to energies.train.* files containing predicted vs reference
        energies for training set. Default: None
    energies_test_files : List[os.PathLike], optional
        Paths to energies.test.* files containing predicted vs reference
        energies for test set. Default: None
    history_data : Dict[str, List[float]], optional
        In-memory training history dictionary (for PyTorch backend).
        Expected keys include 'train_energy_rmse', 'test_energy_rmse',
        'train_energy_mae', 'test_energy_mae', and optionally force
        RMSE and timing data. Default: None
    config : Any, optional
        Training configuration object for metadata. Default: None

    Attributes
    ----------
    path : os.PathLike or None
        Path to training output file (Fortran backend only)
    errors : pd.DataFrame
        Training and test errors per epoch. Columns include:
        - MAE_train: Mean absolute error on training set
        - RMSE_train: Root mean square error on training set
        - MAE_test: Mean absolute error on test set
        - RMSE_test: Root mean square error on test set
        - RMSE_force_train: Force RMSE on training set (PyTorch only)
        - RMSE_force_test: Force RMSE on test set (PyTorch only)
    energies : Energies or None
        Energy predictions vs references (if available)
    timing : pd.DataFrame or None
        Timing information per epoch (PyTorch only). Columns include
        epoch_time, forward_time, backward_time.
    learning_rate : List[float] or None
        Learning rate schedule over epochs (PyTorch only)

    Raises
    ------
    ValueError
        If neither path nor history_data is provided, or if both are provided

    Examples
    --------
    Load training results from Fortran train.x output:

    >>> results = TrainOut('train.out')
    >>> print(results.stats)
    >>> results.plot_training_errors()

    Create from PyTorch training history:

    >>> history = {
    ...     'train_energy_rmse': [0.1, 0.08, 0.06],
    ...     'test_energy_rmse': [0.12, 0.09, 0.07],
    ...     'train_energy_mae': [0.08, 0.06, 0.05],
    ...     'test_energy_mae': [0.09, 0.07, 0.06],
    ... }
    >>> results = TrainOut.from_torch_history(history)
    >>> print(results.stats)

    See Also
    --------
    Energies : Parser for energy prediction files
    """

    def __init__(
        self,
        path: Optional[os.PathLike] = None,
        energies_train_files: Optional[List[os.PathLike]] = None,
        energies_test_files: Optional[List[os.PathLike]] = None,
        history_data: Optional[Dict[str, List[float]]] = None,
        config: Optional[Any] = None,
    ) -> None:
        # Validate inputs
        if path is None and history_data is None:
            raise ValueError(
                "Either 'path' or 'history_data' must be provided"
            )
        if path is not None and history_data is not None:
            raise ValueError(
                "Cannot provide both 'path' and 'history_data'"
            )

        self.config = config

        # Fortran backend path (file-based)
        if path is not None:
            self.path = path
            self.errors = self._read_training_errors()
            self.energies = self._read_energy_files(
                energies_train_files, energies_test_files
            )
            self.timing = None
            self.learning_rate = None
        # PyTorch backend path (in-memory)
        else:
            self.path = None
            self.errors = self._build_errors_from_history(history_data)
            self.energies = None
            self.timing = self._build_timing_from_history(history_data)
            self.learning_rate = history_data.get('learning_rate', None)

    def __str__(self) -> str:
        """Return string representation with training statistics."""
        out = "Training statistics:\n"
        for k, v in self.stats.items():
            out += "  {}: {}\n".format(k, v)
        return out

    def __repr__(self) -> str:
        """Return detailed string representation."""
        n_epochs = len(self.errors)
        source = "file" if self.path else "in-memory"
        return (f"TrainOut(n_epochs={n_epochs}, source={source}, "
                f"has_forces={self.has_force_data})")

    @property
    def has_force_data(self) -> bool:
        """Check if force RMSE data is available."""
        return ('RMSE_force_train' in self.errors.columns or
                'RMSE_force_test' in self.errors.columns)

    def _read_training_errors(self,
                              path: Optional[os.PathLike] = None
                              ) -> pd.DataFrame:
        """
        Read training errors from 'train.x' output file.

        Parameters
        ----------
        path : os.PathLike, optional
            Path to training output file. If None, uses self.path.
            Default: None

        Returns
        -------
        pd.DataFrame
            DataFrame with columns MAE_train, RMSE_train, MAE_test, RMSE_test

        Raises
        ------
        FileNotFoundError
            If the output file cannot be found
        """
        if path is None:
            path = self.path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Training output file not found: {path}")

        errors = []
        with open(path) as fp:
            for line in fp:
                if re.match("^ *[0-9].*<$", line):
                    errors.append([float(a) for a in line.split()[1:-1]])
        errors = np.array(errors)
        return pd.DataFrame(
            data=errors,
            columns=['MAE_train', 'RMSE_train', 'MAE_test', 'RMSE_test'])

    def _build_errors_from_history(
        self,
        history: Dict[str, List[float]]
    ) -> pd.DataFrame:
        """
        Build errors DataFrame from PyTorch training history.

        Parameters
        ----------
        history : Dict[str, List[float]]
            Training history dictionary with keys like 'train_energy_rmse',
            'test_energy_rmse', 'train_force_rmse', etc.

        Returns
        -------
        pd.DataFrame
            DataFrame with training and test errors per epoch
        """
        data: Dict[str, List[float]] = {
            'MAE_train': history.get('train_energy_mae', []),
            'RMSE_train': history.get('train_energy_rmse', []),
            'MAE_test': history.get('test_energy_mae', []),
            'RMSE_test': history.get('test_energy_rmse', []),
        }

        # Add force columns if present
        if 'train_force_rmse' in history:
            data['RMSE_force_train'] = history['train_force_rmse']
        if 'test_force_rmse' in history:
            data['RMSE_force_test'] = history['test_force_rmse']

        return pd.DataFrame(data)

    def _build_timing_from_history(
        self,
        history: Dict[str, List[float]]
    ) -> Optional[pd.DataFrame]:
        """
        Build timing DataFrame from PyTorch training history.

        Parameters
        ----------
        history : Dict[str, List[float]]
            Training history dictionary

        Returns
        -------
        pd.DataFrame or None
            DataFrame with timing data per epoch, or None if not available
        """
        timing_keys = ['epoch_time', 'forward_time', 'backward_time']
        available_keys = [k for k in timing_keys if k in history]

        if not available_keys:
            return None

        data = {k: history[k] for k in available_keys}
        return pd.DataFrame(data)

    def _read_energy_files(
        self,
        energies_train_files: Optional[List[os.PathLike]],
        energies_test_files: Optional[List[os.PathLike]]
    ) -> Optional['Energies']:
        """
        Read energies from 'energies.train.*' and 'energies.test.*' files.

        Parameters
        ----------
        energies_train_files : List[os.PathLike] or None
            List of paths to training energy files
        energies_test_files : List[os.PathLike] or None
            List of paths to test energy files

        Returns
        -------
        Energies or None
            Energies object containing predicted vs reference energies,
            or None if no files provided
        """
        if energies_train_files is not None and len(energies_train_files) > 0:
            has_test_energies = False
            if energies_test_files is not None:
                if len(energies_test_files) != len(energies_train_files):
                    import warnings
                    warnings.warn(
                        "Number of training and test energy files differ. "
                        "The test files will be ignored.",
                        UserWarning
                    )
                else:
                    has_test_energies = True

            if has_test_energies:
                energies = Energies(energies_train_files[0],
                                    energies_test_files[0])
                for f_trn, f_tst in zip(energies_train_files[1:],
                                        energies_test_files[1:]):
                    energies.add_energies(f_trn, f_tst)
            else:
                energies = Energies(energies_train_files[0])
                for f_trn in energies_train_files[1:]:
                    energies.add_energies(f_trn)
            return energies
        return None

    @property
    def stats(self) -> Dict[str, Union[float, int]]:
        """
        Return a dictionary with training statistics.

        Returns
        -------
        Dict[str, Union[float, int]]
            Dictionary containing final and best error metrics:
            - final_MAE_train: Final training MAE
            - final_RMSE_train: Final training RMSE
            - final_MAE_test: Final test MAE
            - final_RMSE_test: Final test RMSE
            - min_RMSE_test: Best (minimum) test RMSE
            - epoch_min_RMSE_test: Epoch number with best test RMSE
            - final_RMSE_force_train: Final force RMSE (if available)
            - final_RMSE_force_test: Final force RMSE (if available)

        Examples
        --------
        >>> results = TrainOut('train.out')
        >>> stats = results.stats
        >>> print(f"Final test RMSE: {stats['final_RMSE_test']:.4f}")
        >>> print(f"Best test RMSE: {stats['min_RMSE_test']:.4f} "
        ...       f"at epoch {stats['epoch_min_RMSE_test']}")
        """
        stats: Dict[str, Union[float, int]] = {
            'final_MAE_train': float(self.errors['MAE_train'].values[-1]),
            'final_RMSE_train': float(self.errors['RMSE_train'].values[-1]),
            'final_MAE_test': float(self.errors['MAE_test'].values[-1]),
            'final_RMSE_test': float(self.errors['RMSE_test'].values[-1]),
            'min_RMSE_test': float(np.min(self.errors['RMSE_test'].values)),
            'epoch_min_RMSE_test': int(
                np.argmin(self.errors['RMSE_test'].values)) + 1
        }

        # Add force statistics if available
        if 'RMSE_force_train' in self.errors.columns:
            stats['final_RMSE_force_train'] = float(
                self.errors['RMSE_force_train'].values[-1]
            )
        if 'RMSE_force_test' in self.errors.columns:
            stats['final_RMSE_force_test'] = float(
                self.errors['RMSE_force_test'].values[-1]
            )
            stats['min_RMSE_force_test'] = float(
                np.min(self.errors['RMSE_force_test'].values)
            )
            stats['epoch_min_RMSE_force_test'] = int(
                np.argmin(self.errors['RMSE_force_test'].values)
            ) + 1

        return stats

    def plot_training_errors(
        self,
        data: Optional[pd.DataFrame] = None,
        outfile: Optional[str] = None
    ) -> None:
        """
        Plot training and test RMSE over epochs.

        Parameters
        ----------
        data : pd.DataFrame, optional
            DataFrame with error data. If None, uses self.errors.
            Default: None
        outfile : str, optional
            Path to save figure. If None, displays interactively.
            Default: None

        Examples
        --------
        >>> results = TrainOut('train.out')
        >>> results.plot_training_errors(outfile='training_errors.png')
        """
        plt.rcParams.update(cfg.read('matplotlib_rc_params'))
        if data is None:
            data = self.errors
        data.plot(y=["RMSE_train", "RMSE_test"], logy=True,
                  xlabel="Epoch", ylabel="RMSE (eV/atom)")
        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')
        plt.show()

    def plot_force_errors(self, outfile: Optional[str] = None) -> None:
        """
        Plot force RMSE over epochs (PyTorch only).

        Parameters
        ----------
        outfile : str, optional
            Path to save figure. If None, displays interactively.
            Default: None

        Raises
        ------
        ValueError
            If force RMSE data is not available

        Examples
        --------
        >>> results = TrainOut.from_torch_history(history)
        >>> results.plot_force_errors(outfile='force_errors.png')
        """
        if not self.has_force_data:
            raise ValueError(
                "No force RMSE data available. Force data only available "
                "for PyTorch training with force_fraction > 0."
            )

        plt.rcParams.update(cfg.read('matplotlib_rc_params'))
        cols = []
        if 'RMSE_force_train' in self.errors.columns:
            cols.append('RMSE_force_train')
        if 'RMSE_force_test' in self.errors.columns:
            cols.append('RMSE_force_test')

        self.errors.plot(y=cols, logy=True,
                         xlabel="Epoch", ylabel="Force RMSE (eV/Å)")
        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')
        plt.show()

    def plot_training_summary(self, outfile: Optional[str] = None) -> None:
        """
        Plot combined summary of energy and force training errors.

        Creates a multi-panel figure showing both energy and force RMSE
        evolution over training epochs.

        Parameters
        ----------
        outfile : str, optional
            Path to save figure. If None, displays interactively.
            Default: None

        Examples
        --------
        >>> results = TrainOut.from_torch_history(history)
        >>> results.plot_training_summary(outfile='training_summary.png')
        """
        plt.rcParams.update(cfg.read('matplotlib_rc_params'))

        n_panels = 2 if self.has_force_data else 1
        fig, axes = plt.subplots(n_panels, 1, figsize=(8, 4*n_panels))

        if n_panels == 1:
            axes = [axes]

        # Energy errors
        self.errors.plot(y=["RMSE_train", "RMSE_test"], logy=True,
                         xlabel="Epoch", ylabel="Energy RMSE (eV/atom)",
                         ax=axes[0])
        axes[0].set_title("Energy Errors")

        # Force errors (if available)
        if self.has_force_data:
            cols = []
            if 'RMSE_force_train' in self.errors.columns:
                cols.append('RMSE_force_train')
            if 'RMSE_force_test' in self.errors.columns:
                cols.append('RMSE_force_test')

            self.errors.plot(y=cols, logy=True,
                             xlabel="Epoch", ylabel="Force RMSE (eV/Å)",
                             ax=axes[1])
            axes[1].set_title("Force Errors")

        plt.tight_layout()
        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')
        plt.show()

    @classmethod
    def from_torch_history(
        cls,
        history: Dict[str, List[float]],
        config: Optional[Any] = None,
    ) -> 'TrainOut':
        """
        Create TrainOut from PyTorch training history.

        Factory method for creating TrainOut objects from in-memory
        training data produced by TorchANNPotential.train().

        Parameters
        ----------
        history : Dict[str, List[float]]
            Training history dictionary with keys like:
            - 'train_energy_rmse': Training energy RMSE per epoch
            - 'test_energy_rmse': Test energy RMSE per epoch
            - 'train_energy_mae': Training energy MAE per epoch
            - 'test_energy_mae': Test energy MAE per epoch
            - 'train_force_rmse': Training force RMSE (optional)
            - 'test_force_rmse': Test force RMSE (optional)
            - 'learning_rate': Learning rate schedule (optional)
            - 'epoch_time': Time per epoch (optional)
        config : TorchTrainingConfig, optional
            Training configuration for metadata. Default: None

        Returns
        -------
        TrainOut
            TrainOut instance populated with PyTorch training data

        Examples
        --------
        >>> from aenet.torch_training import TorchANNPotential
        >>> trainer = TorchANNPotential(arch, descriptor)
        >>> results = trainer.train(structures, config=config)
        >>> print(results.stats)
        >>> results.plot_training_summary()
        """
        return cls(history_data=history, config=config)


class Energies(object):
    """
    Parser and representation of 'energies.train.*' and
    'energies.test.*' files.

    These files are generated by aenet's train.x and contain predicted vs
    reference energies for each structure in the training and test sets.

    Parameters
    ----------
    path_train : os.PathLike
        Path to the training energy file (e.g., 'energies.train.0')
    path_test : os.PathLike, optional
        Path to the test energy file (e.g., 'energies.test.0').
        Default: None

    Attributes
    ----------
    path_train : os.PathLike
        Path to training energy file
    path_test : os.PathLike or None
        Path to test energy file
    energies_train : pd.DataFrame
        Training set energies with columns from the file header
    energies_test : pd.DataFrame or None
        Test set energies (if test file provided)

    Examples
    --------
    >>> energies = Energies('energies.train.0', 'energies.test.0')
    >>> energies.plot_correlation(outfile='correlation.png')
    """

    def __init__(
        self,
        path_train: os.PathLike,
        path_test: Optional[os.PathLike] = None
    ) -> None:
        self.path_train = path_train
        self.path_test = path_test
        with open(path_train) as fp:
            columns = fp.readline()
        columns = columns.replace("Cost Func", "Cost-Func")
        columns = columns.split()
        self._columns = columns
        self.energies_train = pd.read_csv(path_train,
                                          sep=r'\s+',
                                          skiprows=0, header=0,
                                          names=columns)
        if path_test is not None:
            self.energies_test = pd.read_csv(path_test,
                                             sep=r'\s+',
                                             skiprows=0, header=0,
                                             names=columns)
        else:
            self.energies_test = None

    def __str__(self) -> str:
        """Return string representation of energy data."""
        n_train = len(self.energies_train)
        n_test = (len(self.energies_test)
                  if self.energies_test is not None else 0)
        return (f"Energies(n_train={n_train}, n_test={n_test})")

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return self.__str__()

    def add_energies(
        self,
        path_train: os.PathLike,
        path_test: Optional[os.PathLike] = None
    ) -> None:
        """
        Add energies from additional 'energies.train.*' and
        'energies.test.*' files.

        This is useful when training produces multiple energy files that need
        to be concatenated for analysis.

        Parameters
        ----------
        path_train : os.PathLike
            Path to additional training energy file
        path_test : os.PathLike, optional
            Path to additional test energy file. Default: None

        Examples
        --------
        >>> energies = Energies('energies.train.0')
        >>> energies.add_energies('energies.train.1')
        >>> energies.add_energies('energies.train.2')
        """
        energies_train_new = pd.read_csv(path_train,
                                         sep=r'\s+',
                                         skiprows=0, header=0,
                                         names=self._columns)
        self.energies_train = pd.concat(
            [self.energies_train, energies_train_new],
            ignore_index=True)
        if path_test is not None:
            energies_test_new = pd.read_csv(path_test,
                                            sep=r'\s+',
                                            skiprows=0, header=0,
                                            names=self._columns)
            self.energies_test = pd.concat(
                [self.energies_test, energies_test_new],
                ignore_index=True)

    def plot_correlation(
        self,
        outfile: Optional[str] = None,
        E_min: Optional[float] = None,
        E_max: Optional[float] = None
    ) -> None:
        """
        Plot correlation between ANN-predicted and reference energies.

        Creates a scatter plot showing training and test set predictions
        vs reference values with a diagonal line indicating perfect agreement.

        Parameters
        ----------
        outfile : str, optional
            Path to save figure. If None, displays interactively.
            Default: None
        E_min : float, optional
            Minimum energy for plot axes. If None, uses data minimum.
            Default: None
        E_max : float, optional
            Maximum energy for plot axes. If None, uses data maximum.
            Default: None

        Examples
        --------
        >>> energies = Energies('energies.train.0', 'energies.test.0')
        >>> energies.plot_correlation(outfile='correlation.png',
        ...                           E_min=-10, E_max=-5)
        """
        plt.rcParams.update(cfg.read('matplotlib_rc_params'))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.scatter(self.energies_train["ANN(eV/atom)"].values,
                    self.energies_train["Ref(eV/atom)"].values,
                    label="Training Set")
        if self.energies_test is not None:
            plt.scatter(self.energies_test["ANN(eV/atom)"].values,
                        self.energies_test["Ref(eV/atom)"].values,
                        label="Validation Set")
        x0 = np.min(self.energies_train["Ref(eV/atom)"].values)
        x1 = np.max(self.energies_train["Ref(eV/atom)"].values)
        if E_min is not None:
            x0 = E_min
        if E_max is not None:
            x1 = E_max
        x = np.linspace(x0, x1, 100)
        plt.plot(x, x, color="black", label="")
        plt.xlim([x0, x1])
        plt.ylim([x0, x1])
        plt.xlabel("ANN (eV/atom)")
        plt.ylabel("Reference (eV/atom)")
        plt.legend(loc="upper left")
        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')
        plt.show()
