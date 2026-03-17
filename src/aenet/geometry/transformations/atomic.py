"""Atomic (coordinate) transformations.

This module contains transformations that primarily modify *atomic
coordinates* while keeping the unit cell unchanged.

Notes
-----
These transformations operate in Cartesian coordinates as stored on
:class:`~aenet.geometry.structure.AtomicStructure`.
"""

import logging
import warnings
from collections.abc import Iterator
from typing import Optional, Union

import numpy as np
from scipy.optimize import minimize

from ..structure import AtomicStructure
from .base import TransformationABC

logger = logging.getLogger(__name__)


class AtomDisplacementTransformation(TransformationABC):
    """Displace each atom in Cartesian x, y, z directions.

    This deterministic transformation generates ``3N`` structures, where
    ``N`` is the number of atoms. Each structure has one atom displaced
    by a fixed amount in one Cartesian direction.

    Parameters
    ----------
    displacement : float, optional
        Magnitude of displacement in Angstroms (default: 0.1)

    Notes
    -----
    This transformation is useful for finite-difference derivatives
    (forces, Hessians, etc.) or generating local perturbations around a
    reference structure.
    """

    def __init__(self, displacement: float = 0.1):
        if displacement <= 0:
            raise ValueError(
                f"Displacement must be positive, got {displacement}"
            )
        self.displacement = displacement
        logger.info(
            "AtomDisplacementTransformation initialized with "
            "displacement=%s Ang",
            displacement,
        )

    def apply_transformation(
        self,
        structure: AtomicStructure,
        **kwargs
    ) -> Iterator[AtomicStructure]:
        """Apply displacement transformation to structure.

        Yields 3N structures, one for each atom displaced in each direction.
        """
        x_disp = np.array([1.0, 0.0, 0.0]) * self.displacement
        y_disp = np.array([0.0, 1.0, 0.0]) * self.displacement
        z_disp = np.array([0.0, 0.0, 1.0]) * self.displacement

        for atom_idx in range(structure.natoms):
            for direction, disp_vector in enumerate([x_disp, y_disp, z_disp]):
                displaced = structure.copy()
                displaced.coords[-1][atom_idx] += disp_vector

                logger.debug(
                    "Displaced atom %d in direction %s by %.6f Ang",
                    atom_idx,
                    "xyz"[direction],
                    self.displacement,
                )
                yield displaced

        logger.info(
            "Generated %d displaced structures",
            3 * structure.natoms,
        )


class RandomDisplacementTransformation(TransformationABC):
    """Generate random atomic displacement vectors.

    This stochastic transformation creates structures with random atomic
    displacements. It supports two modes:

    1. ``orthonormalize=True`` (default): generates an orthonormal basis
       of displacement directions in the full ``3N`` space (optionally
       with the 3 translational modes removed). This is often useful to
       generate a compact, non-redundant set of perturbations.

    2. ``orthonormalize=False``: draws independent random displacement
       patterns. This is useful when you want many samples, without
       imposing orthogonality constraints.

    Purpose
    -------
    This transformation is mainly intended for structure-space sampling
    (e.g., active learning) and for generating training data for
    interatomic potentials.

    Parameters
    ----------
    rms : float, optional
        Target RMS displacement in Angstroms (default: 0.1)
    max_structures : int, optional
        Maximum number of structures to generate. If None, defaults to
        ``3N-3`` when ``remove_translations=True`` and to ``3N``
        otherwise.
    random_state : int or numpy.random.Generator, optional
        Random seed or generator for reproducibility.
    orthonormalize : bool, optional
        If True, generate orthonormal displacement vectors using QR.
    remove_translations : bool, optional
        If True, remove the 3 translational modes (fixed center of mass).

    """

    RMS_TOLERANCE = 1e-6

    def __init__(
        self,
        rms: float = 0.1,
        max_structures: Optional[int] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        orthonormalize: bool = True,
        remove_translations: bool = True,
    ):
        if rms <= 0:
            raise ValueError(f"RMS must be positive, got {rms}")
        if max_structures is not None and max_structures <= 0:
            raise ValueError(
                f"max_structures must be positive, got {max_structures}"
            )

        self.rms = rms
        self.max_structures = max_structures
        self.orthonormalize = orthonormalize
        self.remove_translations = remove_translations

        if isinstance(random_state, np.random.Generator):
            self.rng = random_state
        elif random_state is not None:
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = np.random.default_rng()

        logger.info(
            "RandomDisplacementTransformation initialized: rms=%s Ang, "
            "max_structures=%s, orthonormalize=%s, remove_translations=%s",
            rms,
            max_structures if max_structures is not None else "default",
            orthonormalize,
            remove_translations,
        )

    def apply_transformation(
        self,
        structure: AtomicStructure,
        **kwargs
    ) -> Iterator[AtomicStructure]:
        """Apply random displacement transformation to structure.

        Yields structures with random atomic displacements.
        """
        natoms = structure.natoms
        dim = 3 * natoms

        if self.max_structures is None:
            max_structures = dim - 3 if self.remove_translations else dim
        else:
            max_structures = self.max_structures

        if self.orthonormalize:
            max_available = dim - 3 if self.remove_translations else dim
            n_structures = min(max_structures, max_available)
            if max_structures > max_available:
                warnings.warn(
                    f"Requested {max_structures} structures but only "
                    f"{max_available} orthonormal vectors available. "
                    f"Generating {n_structures} structures.",
                    RuntimeWarning,
                )

            random_matrix = self.rng.standard_normal((dim, n_structures))
            Q, _ = np.linalg.qr(random_matrix)

            if self.remove_translations:
                Q = self._remove_translational_modes(Q, natoms)

            for i in range(Q.shape[1]):
                displacement = Q[:, i].reshape(natoms, 3)

                current_rms = np.sqrt(np.mean(displacement ** 2))
                if current_rms <= 0:
                    warnings.warn(
                        f"Zero displacement vector at index {i}",
                        RuntimeWarning,
                    )
                    continue
                displacement = displacement * (self.rms / current_rms)

                achieved_rms = np.sqrt(np.mean(displacement ** 2))
                if abs(achieved_rms - self.rms) > self.RMS_TOLERANCE:
                    warnings.warn(
                        "RMS normalization error: "
                        f"target={self.rms:.6f}, achieved={achieved_rms:.6f}",
                        RuntimeWarning,
                    )

                displaced = structure.copy()
                displaced.coords[-1] = structure.coords[-1] + displacement
                yield displaced

        else:
            n_structures = max_structures
            displacements = self.rng.standard_normal((n_structures, natoms, 3))

            if self.remove_translations:
                com_disp = np.mean(displacements, axis=1, keepdims=True)
                displacements = displacements - com_disp

            rms_values = np.sqrt(
                np.mean(displacements ** 2, axis=(1, 2), keepdims=True)
            )
            rms_values = np.where(rms_values == 0.0, 1.0, rms_values)
            displacements = displacements * (self.rms / rms_values)

            for i in range(n_structures):
                displacement = displacements[i]

                achieved_rms = np.sqrt(np.mean(displacement ** 2))
                if abs(achieved_rms - self.rms) > self.RMS_TOLERANCE:
                    warnings.warn(
                        "RMS normalization error: "
                        f"target={self.rms:.6f}, achieved={achieved_rms:.6f}",
                        RuntimeWarning,
                    )

                displaced = structure.copy()
                displaced.coords[-1] = structure.coords[-1] + displacement
                yield displaced

    def _remove_translational_modes(
            self, Q: np.ndarray, natoms: int) -> np.ndarray:
        """
        Remove translational modes from orthonormal basis.

        This method projects out the 3 uniform translation modes
        (all atoms moving together in x, y, or z) from the basis
        vectors and re-orthonormalizes.

        Parameters
        ----------
        Q : ndarray of shape (3*natoms, n_vectors)
            Orthonormal basis vectors (columns)
        natoms : int
            Number of atoms

        Returns
        -------
        ndarray of shape (3*natoms, n_vectors)
            Basis with translational modes removed and re-orthonormalized
        """
        dim = 3 * natoms

        # Define translational modes: uniform displacement in x, y, z
        t1 = np.zeros(dim)
        t2 = np.zeros(dim)
        t3 = np.zeros(dim)

        for i in range(natoms):
            t1[3*i] = 1.0      # x-translation
            t2[3*i + 1] = 1.0  # y-translation
            t3[3*i + 2] = 1.0  # z-translation

        # Normalize translational modes
        t1 = t1 / np.linalg.norm(t1)
        t2 = t2 / np.linalg.norm(t2)
        t3 = t3 / np.linalg.norm(t3)

        # Project out translational modes from each column of Q
        Q_proj = Q.copy()
        for i in range(Q.shape[1]):
            v = Q[:, i]
            # Remove projection onto each translational mode
            v = v - np.dot(v, t1) * t1
            v = v - np.dot(v, t2) * t2
            v = v - np.dot(v, t3) * t3
            # Re-normalize
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                Q_proj[:, i] = v / norm
            else:
                # Vector is parallel to translational modes, set to zero
                Q_proj[:, i] = 0.0

        # Remove zero columns (vectors that were purely translational)
        non_zero_cols = [
            i for i in range(Q_proj.shape[1])
            if np.linalg.norm(Q_proj[:, i]) > 1e-10
        ]
        Q_proj = Q_proj[:, non_zero_cols]

        # If nothing remains, return as-is (empty)
        if Q_proj.shape[1] == 0:
            logger.debug(
                f"Removed translational modes: {Q.shape[1]} -> 0 vectors"
            )
            return Q_proj

        # Re-orthonormalize columns to restore mutual orthogonality
        Q_orth, _ = np.linalg.qr(Q_proj, mode='reduced')

        logger.debug(
            f"Removed translational modes: {Q.shape[1]} -> "
            f"{Q_orth.shape[1]} vectors (re-orthonormalized)"
        )

        return Q_orth


class DOptimalDisplacementTransformation(TransformationABC):
    """Generate D-optimal (maximally diverse) atomic displacements.

    This stochastic transformation generates a *fixed number* of
    displaced structures around a reference configuration. The ensemble
    of displacements is optimized to be maximally diverse by maximizing
    the log-determinant of the (regularized) covariance matrix of the
    displacement patterns (D-optimal design criterion).

    Purpose
    -------
    The operation is a set of coordinate perturbations at fixed cell.
    The D-optimality objective seeks a set of displacement patterns that
    span the local configuration space as widely as possible for a
    given number of samples.

    Parameters
    ----------
    rms : float, optional
        Target RMS displacement per atom in Angstroms (default: 0.1).
    n_structures : int, optional
        Number of displaced structures to generate (>= 2).
    max_iter : int, optional
        Maximum number of L-BFGS-B iterations (default: 200).
    learning_rate : float, optional
        (Kept for backward compatibility with earlier projected-gradient
        implementations; currently unused by the SciPy optimizer.)
    tol : float, optional
        Optimizer tolerance (default: 1e-4).
    logdet_regularization : float, optional
        Regularization parameter epsilon for covariance (default: 1e-6).
    random_state : int or numpy.random.Generator, optional
        Random seed or generator for reproducibility.
    remove_translations : bool, optional
        If True, remove COM translation per pattern.
    enforce_zero_mean : bool, optional
        If True, enforce zero mean displacement across ensemble.

    """

    def __init__(
        self,
        rms: float = 0.1,
        n_structures: int = 10,
        max_iter: int = 200,
        learning_rate: float = 0.1,
        tol: float = 1e-4,
        logdet_regularization: float = 1e-6,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        remove_translations: bool = True,
        enforce_zero_mean: bool = True,
        verbose: bool = False,
    ):
        if rms <= 0:
            raise ValueError(f"RMS must be positive, got {rms}")
        if n_structures < 2:
            raise ValueError(
                f"n_structures must be at least 2, got {n_structures}"
            )
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}")
        if learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {learning_rate}"
            )
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}")
        if logdet_regularization <= 0:
            raise ValueError(
                "logdet_regularization must be positive, got "
                f"{logdet_regularization}"
            )

        self.rms = rms
        self.n_structures = n_structures
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.logdet_regularization = logdet_regularization
        self.remove_translations = remove_translations
        self.enforce_zero_mean = enforce_zero_mean
        self.verbose = verbose

        if isinstance(random_state, np.random.Generator):
            self.rng = random_state
        elif random_state is not None:
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = np.random.default_rng()

    def apply_transformation(
        self,
        structure: AtomicStructure,
        **kwargs
    ) -> Iterator[AtomicStructure]:
        """Apply D-optimal displacement transformation to structure.

        Yields n_structures structures with maximally diverse displacements.
        """
        natoms = structure.natoms
        dim_full = 3 * natoms
        n_p = self.n_structures

        rand_transform = RandomDisplacementTransformation(
            rms=self.rms,
            max_structures=n_p,
            random_state=self.rng,
            orthonormalize=False,
            remove_translations=self.remove_translations,
        )
        rand_structures = list(rand_transform.apply_transformation(structure))

        X_rand = np.vstack([
            (s.coords[-1] - structure.coords[-1]).reshape(dim_full)
            for s in rand_structures
        ])

        J_rand, _ = self._objective_and_grad(X_rand)

        X0 = self._project_ensemble(X_rand, natoms)
        theta0 = self._X_to_theta(X0)

        def _obj(theta: np.ndarray) -> tuple:
            return self._scipy_objective(theta, natoms=natoms, n_p=n_p)

        def fun(theta: np.ndarray) -> float:
            f, _ = _obj(theta)
            return f

        def jac(theta: np.ndarray) -> np.ndarray:
            _, g = _obj(theta)
            return g

        result = minimize(
            fun=fun,
            x0=theta0,
            jac=jac,
            method="L-BFGS-B",
            options={
                "maxiter": self.max_iter,
                "ftol": self.tol,
            },
        )

        X_opt = self._theta_to_X(result.x, n_p, dim_full)
        X_opt = self._project_ensemble(X_opt, natoms)

        J_opt, _ = self._objective_and_grad(X_opt)

        if np.isneginf(J_opt) or (J_opt + 1e-8 < J_rand):
            X_final = self._project_ensemble(X_rand, natoms)
        else:
            X_final = X_opt

        def _gen() -> Iterator[AtomicStructure]:
            for i in range(n_p):
                disp_coords = self._flat_to_coords(X_final[i], natoms)
                displaced = structure.copy()
                displaced.coords[-1] = structure.coords[-1] + disp_coords
                yield displaced

        return _gen()

    # --- helper methods (unchanged from original implementation) ---

    def _theta_to_X(self, theta: np.ndarray, n_p: int, d: int) -> np.ndarray:
        return theta.reshape(n_p, d)

    def _X_to_theta(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(-1)

    def _scipy_objective(
            self, theta: np.ndarray, natoms: int, n_p: int) -> tuple:
        d = 3 * natoms
        X = self._theta_to_X(theta, n_p, d)
        X_proj = self._project_ensemble(X, natoms)
        J, grad_X = self._objective_and_grad(X_proj)
        if np.isneginf(J):
            return np.inf, np.zeros_like(theta)
        return -J, -self._X_to_theta(grad_X)

    @staticmethod
    def _flat_to_coords(flat: np.ndarray, natoms: int) -> np.ndarray:
        return flat.reshape(natoms, 3)

    def _project_com_free(self, X: np.ndarray, natoms: int) -> np.ndarray:
        n_p, dim_full = X.shape
        if dim_full != 3 * natoms:
            raise ValueError(
                f"Expected dim_full=3N={3 * natoms}, got {dim_full}"
            )
        disp = X.reshape(n_p, natoms, 3)
        com = disp.mean(axis=1, keepdims=True)
        disp_no_com = disp - com
        return disp_no_com.reshape(n_p, dim_full)

    @staticmethod
    def _enforce_zero_mean(X: np.ndarray) -> np.ndarray:
        mu = X.mean(axis=0, keepdims=True)
        return X - mu

    def _enforce_rms(self, X: np.ndarray, natoms: int) -> np.ndarray:
        target_norm = np.sqrt(3 * natoms) * self.rms
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        scale = target_norm / norms
        if np.allclose(scale, 1.0, rtol=1e-12, atol=0.0):
            return X
        return X * scale

    def _project_ensemble(self, X: np.ndarray, natoms: int) -> np.ndarray:
        if self.remove_translations:
            X = self._project_com_free(X, natoms)
        if self.enforce_zero_mean:
            X = self._enforce_zero_mean(X)
        X = self._enforce_rms(X, natoms)
        if self.enforce_zero_mean:
            X = self._enforce_zero_mean(X)
        return X

    def _objective_and_grad(self, X: np.ndarray) -> tuple:
        n_p, d = X.shape
        Xc = X - X.mean(axis=0, keepdims=True)

        if n_p <= 1:
            return -np.inf, np.zeros_like(X)

        eps = self.logdet_regularization
        XXt = Xc @ Xc.T / float(n_p - 1)
        M = XXt + eps * np.eye(n_p)

        sign, logdet_M = np.linalg.slogdet(M)
        if sign <= 0:
            return -np.inf, np.zeros_like(X)

        J = (d - n_p) * np.log(eps) + logdet_M

        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return -np.inf, np.zeros_like(X)

        grad_Xc = (2.0 / (eps * float(n_p - 1))) * (
            Xc - (Xc @ Xc.T @ M_inv @ Xc) / float(n_p - 1)
        )

        grad_X = grad_Xc
        if self.enforce_zero_mean:
            grad_X = grad_X - grad_X.mean(axis=0, keepdims=True)

        return J, grad_X

    def _remove_translational_modes(
            self, Q: np.ndarray, natoms: int) -> np.ndarray:
        dim = 3 * natoms
        t1 = np.zeros(dim)
        t2 = np.zeros(dim)
        t3 = np.zeros(dim)
        for i in range(natoms):
            t1[3 * i] = 1.0
            t2[3 * i + 1] = 1.0
            t3[3 * i + 2] = 1.0
        t1 = t1 / np.linalg.norm(t1)
        t2 = t2 / np.linalg.norm(t2)
        t3 = t3 / np.linalg.norm(t3)

        Q_proj = Q.copy()
        for i in range(Q.shape[1]):
            v = Q[:, i]
            v = v - np.dot(v, t1) * t1
            v = v - np.dot(v, t2) * t2
            v = v - np.dot(v, t3) * t3
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                Q_proj[:, i] = v / norm
            else:
                Q_proj[:, i] = 0.0

        non_zero_cols = [
            i for i in range(Q_proj.shape[1])
            if np.linalg.norm(Q_proj[:, i]) > 1e-10
        ]
        Q_proj = Q_proj[:, non_zero_cols]
        if Q_proj.shape[1] == 0:
            return Q_proj

        Q_orth, _ = np.linalg.qr(Q_proj, mode="reduced")
        return Q_orth
