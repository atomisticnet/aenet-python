"""Strain transformations.

This module contains transformations that modify the unit cell by
applying *strain* or *deformation gradients*.

Important distinction
---------------------
In computational materials science there are multiple conventions to
parameterize deformation.

- Some deformations are chosen for *elastic constants* calculations and
  follow standard textbook strain patterns (often volume-conserving).

- Others are more general lattice perturbations useful for *structure-
  space sampling*.

The transformations here try to be explicit about their physical or
engineering meaning in the respective docstrings.
"""

import logging
import warnings
from collections.abc import Iterator

import numpy as np

from .. import utils as geom_utils
from ..structure import AtomicStructure
from .base import TransformationABC

logger = logging.getLogger(__name__)


class IsovolumetricStrainTransformation(TransformationABC):
    """Volume-preserving *uniaxial* strain.

    This transformation scales *one* lattice direction by a factor ``s``
    while scaling the other two directions by ``s**(-1/2)`` to preserve
    the cell volume. Fractional coordinates are preserved, Cartesian
    coordinates are rebuilt from the deformed cell, and copied
    energy/force labels are cleared.

    Physical/engineering meaning
    ----------------------------
    This is a constrained deformation (volume fixed by construction).
    It is **not** the standard deformation used for Young's modulus,
    because Young's modulus corresponds to uniaxial stress / stress-free
    transverse directions and generally changes volume.

    This transformation is useful for *structure-space sampling* when
    you want to explore anisotropic cell shapes without changing volume.

    Parameters
    ----------
    direction : int
        Direction to strain (1=a, 2=b, 3=c)
    len_min : float
        Minimum scaling factor for the strained direction
    len_max : float
        Maximum scaling factor for the strained direction
    steps : int
        Number of strain steps
    """

    VOLUME_TOLERANCE = 1e-5

    def __init__(self, direction: int, len_min: float,
                 len_max: float, steps: int):
        if direction not in (1, 2, 3):
            raise ValueError(f"direction must be 1, 2, or 3, got {direction}")
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        if len_min <= 0 or len_max <= 0:
            raise ValueError("scaling factors must be positive")
        if len_min >= len_max:
            raise ValueError("len_min must be less than len_max")

        self.direction = direction
        self.len_min = len_min
        self.len_max = len_max
        self.steps = steps

    def apply_transformation(
        self,
        structure: AtomicStructure,
        **kwargs,
    ) -> Iterator[AtomicStructure]:
        """Apply isovolumetric strain to structure.

        Yields structures with volume-conserving strain.
        """
        if not structure.pbc:
            raise ValueError(
                "IsovolumetricStrainTransformation requires periodic structure"
            )

        def _gen() -> Iterator[AtomicStructure]:
            s_values = np.linspace(self.len_min, self.len_max, self.steps)
            original_volume = structure.cellvolume()

            for s in s_values:
                s_orth = (1.0 / s) ** 0.5

                if self.direction == 1:
                    scaling = np.diag([s, s_orth, s_orth])
                elif self.direction == 2:
                    scaling = np.diag([s_orth, s, s_orth])
                else:
                    scaling = np.diag([s_orth, s_orth, s])

                strained = structure.copy()
                strained.update_cell(
                    structure.avec[-1] @ scaling,
                    preserve="fractional",
                )

                new_volume = strained.cellvolume()
                if abs(new_volume - original_volume) > self.VOLUME_TOLERANCE:
                    warnings.warn(
                        f"Volume not conserved: {original_volume:.6f} -> "
                        f"{new_volume:.6f}",
                        RuntimeWarning,
                    )

                yield strained

        return _gen()


class UniaxialStrainTransformation(TransformationABC):
    """Uniaxial strain (simple scaling of one lattice direction).

    This transformation scales one lattice direction by a factor ``s``
    and leaves the other two directions unchanged. Fractional
    coordinates are preserved, Cartesian coordinates are rebuilt from
    the deformed cell, and copied energy/force labels are cleared.

    Physical/engineering meaning
    ----------------------------
    This is the simplest *uniaxial strain* deformation. It is often used
    as a starting point to compute directional stiffness / Young's
    modulus, but note:

    - Young's modulus corresponds to *uniaxial stress* with stress-free
      transverse directions. Accurately reproducing that condition may
      require relaxing the transverse cell vectors (and/or internal
      coordinates) at each applied strain.
    - This transformation alone therefore gives you a strained cell, but
      not necessarily the fully relaxed response.

    Parameters
    ----------
    direction : int
        Direction to strain (1=a, 2=b, 3=c)
    len_min : float
        Minimum scaling factor for the strained direction
    len_max : float
        Maximum scaling factor for the strained direction
    steps : int
        Number of strain steps
    """

    def __init__(self, direction: int, len_min: float,
                 len_max: float, steps: int):
        if direction not in (1, 2, 3):
            raise ValueError(f"direction must be 1, 2, or 3, got {direction}")
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        if len_min <= 0 or len_max <= 0:
            raise ValueError("scaling factors must be positive")
        if len_min >= len_max:
            raise ValueError("len_min must be less than len_max")

        self.direction = direction
        self.len_min = len_min
        self.len_max = len_max
        self.steps = steps

    def apply_transformation(
        self,
        structure: AtomicStructure,
        **kwargs,
    ) -> Iterator[AtomicStructure]:
        """Apply uniaxial strain to structure.

        Yields structures with strain in one direction only.
        """
        if not structure.pbc:
            raise ValueError(
                "UniaxialStrainTransformation requires periodic structure")

        def _gen() -> Iterator[AtomicStructure]:
            s_values = np.linspace(self.len_min, self.len_max, self.steps)

            for s in s_values:
                if self.direction == 1:
                    scaling = np.diag([s, 1.0, 1.0])
                elif self.direction == 2:
                    scaling = np.diag([1.0, s, 1.0])
                else:
                    scaling = np.diag([1.0, 1.0, s])

                strained = structure.copy()
                strained.update_cell(
                    structure.avec[-1] @ scaling,
                    preserve="fractional",
                )
                yield strained

        return _gen()


class ShearStrainTransformation(TransformationABC):
    r"""Simple shear (volume-preserving by construction).

    This transformation applies a *simple shear* deformation gradient
    ``F`` with ``det(F)=1``:

    - direction=1: xy shear, F[0, 1] = shear
    - direction=2: xz shear, F[0, 2] = shear
    - direction=3: yz shear, F[1, 2] = shear

    Physical/engineering meaning
    ----------------------------
    Simple shear deforms the cell shape without changing volume
    (determinant of the deformation gradient is 1). Fractional
    coordinates are preserved, Cartesian coordinates are rebuilt from
    the deformed cell, and copied energy/force labels are cleared.

    For small strains in cubic materials, this can be used to extract a
    shear elastic constant (often associated with ``C44`` for xz/yz
    shear). For non-cubic crystals, the relation is more complex.

    Parameters
    ----------
    direction : int
        Shear plane (1=xy, 2=xz, 3=yz)
    shear_min : float
        Minimum shear value (deformation gradient component)
    shear_max : float
        Maximum shear value (deformation gradient component)
    steps : int
        Number of shear steps

    Notes
    -----
    The ``shear_min`` and ``shear_max`` parameters specify off-diagonal
    components of the deformation gradient F (not strain tensor components).

    - For structure-space sampling, a typical range is -0.2 to +0.2. This
      explores moderate cell shape variations while maintaining reasonable
      atomic configurations. Larger values
      (:math:`|\mathrm{shear}| > 0.3`) can lead to highly skewed cells.
    - For elastic constant calculations, stay in the linear elastic regime
      around -0.05 to +0.05 and use at least 5-7 points spanning positive
      and negative values.
    - There is no hard mathematical limit because :math:`\det(F)=1` for all
      finite shear values. Practical limits depend on structure stability and
      the desired sampling regime.
    """

    DETERMINANT_TOLERANCE = 1e-3

    def __init__(self, direction: int, shear_min: float,
                 shear_max: float, steps: int):
        if direction not in (1, 2, 3):
            raise ValueError(f"direction must be 1, 2, or 3, got {direction}")
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        if shear_min >= shear_max:
            raise ValueError("shear_min must be less than shear_max")

        self.direction = direction
        self.shear_min = shear_min
        self.shear_max = shear_max
        self.steps = steps

    def apply_transformation(
        self,
        structure: AtomicStructure,
        **kwargs,
    ) -> Iterator[AtomicStructure]:
        """Apply volume-conserving shear strain to structure.

        Yields structures with simple shear deformation.
        """
        if not structure.pbc:
            raise ValueError(
                "ShearStrainTransformation requires periodic structure")

        def _gen() -> Iterator[AtomicStructure]:
            shear_values = np.linspace(
                self.shear_min, self.shear_max, self.steps)

            for shear in shear_values:
                F = np.eye(3)
                if self.direction == 1:
                    F[0, 1] = shear
                elif self.direction == 2:
                    F[0, 2] = shear
                else:
                    F[1, 2] = shear

                sheared = structure.copy()
                sheared.update_cell(
                    structure.avec[-1] @ F,
                    preserve="fractional",
                )

                det = np.linalg.det(F)
                if abs(det - 1.0) > self.DETERMINANT_TOLERANCE:
                    warnings.warn(
                        f"Determinant not 1: det={det:.6f}",
                        RuntimeWarning,
                    )

                yield sheared

        return _gen()


class OrthorhombicStrainTransformation(TransformationABC):
    r"""Volume-conserving orthorhombic strain (elastic constant pattern).

    This transformation implements the *volume-conserving orthorhombic
    strain* commonly used for extracting the cubic elastic constant
    combination ``C11 - C12``.

    It is based on :func:`aenet.geometry.utils.strain_orthorhombic`.

    For the default direction mapping (direction=1), the applied strain
    tensor has:

    - \epsilon_xx =  e
    - \epsilon_yy = -e
    - \epsilon_zz = e^2/(1-e^2)

    with other components 0.

    Physical/engineering meaning
    ----------------------------
    This is not a generic uniaxial strain; it is a *specific* strain path
    chosen so that volume is conserved and the elastic energy depends on
    ``C11 - C12`` in a simple way (for cubic crystals). Fractional
    coordinates are preserved, Cartesian coordinates are rebuilt from
    the deformed cell, and copied energy/force labels are cleared.

    Parameters
    ----------
    direction : int
        Which axis plays the role of "x" in the above definition
        (1,2,3). The second axis is taken as the next cyclic axis.
    e_min, e_max : float
        Range of e values (dimensionless strain component)
    steps : int
        Number of strain steps

    Notes
    -----
    The ``e_min`` and ``e_max`` parameters specify the dimensionless strain
    component ε_xx in the strain tensor. Note the formula for the compensating
    strain: ε_zz = e²/(1-e²).

    - The formulation has a singularity at :math:`|e| = 1`, where
      :math:`\epsilon_{zz}` diverges. In practice, require
      :math:`|e_{\min}| < 1` and :math:`|e_{\max}| < 1`.
    - For structure-space sampling, a typical range is -0.15 to +0.15.
      Staying well below :math:`|e| = 0.5` avoids extreme cell
      deformations, and values above :math:`|e| > 0.3` may already be
      unstable for some structures.
    - For elastic constant calculations targeting ``C11 - C12``, stay in the
      linear regime around -0.03 to +0.03, use at least 5-7 points with both
      positive and negative strain values, and ensure :math:`|e| \ll 1`.
    """

    def __init__(self, direction: int, e_min: float, e_max: float, steps: int):
        if direction not in (1, 2, 3):
            raise ValueError(f"direction must be 1, 2, or 3, got {direction}")
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        if e_min >= e_max:
            raise ValueError("e_min must be less than e_max")

        self.direction = direction
        self.e_min = e_min
        self.e_max = e_max
        self.steps = steps

    def apply_transformation(
        self,
        structure: AtomicStructure,
        **kwargs,
    ) -> Iterator[AtomicStructure]:
        """Apply orthorhombic strain pattern to structure.

        Yields structures with volume-conserving orthorhombic strain.
        """
        if not structure.pbc:
            raise ValueError(
                "OrthorhombicStrainTransformation requires periodic structure"
            )

        def _gen() -> Iterator[AtomicStructure]:
            e_values = np.linspace(self.e_min, self.e_max, self.steps)

            for e in e_values:
                # Apply strain in a canonical xyz frame using utils helper
                avec0 = structure.avec[-1]

                # Implement direction by cyclic permutation of axes.
                # direction=1 -> (x,y,z) = (0,1,2)
                # direction=2 -> (x,y,z) = (1,2,0)
                # direction=3 -> (x,y,z) = (2,0,1)
                perm = {
                    1: (0, 1, 2),
                    2: (1, 2, 0),
                    3: (2, 0, 1),
                }[self.direction]

                # Permute to canonical
                A_perm = avec0[list(perm), :]
                A_strained = geom_utils.strain_orthorhombic(A_perm, e)

                # Permute back
                invperm = np.argsort(np.array(perm))
                A_back = A_strained[invperm, :]

                strained = structure.copy()
                strained.update_cell(A_back, preserve="fractional")
                yield strained

        return _gen()


class MonoclinicStrainTransformation(TransformationABC):
    r"""Volume-conserving monoclinic strain (elastic constant pattern).

    This transformation implements a standard *volume-conserving
    monoclinic* strain path used to extract a shear elastic constant
    (often ``C44`` in cubic crystals).

    It is based on :func:`aenet.geometry.utils.strain_monoclinic`.

    Physical/engineering meaning
    ----------------------------
    In contrast to :class:`ShearStrainTransformation` (simple shear
    deformation gradient), this uses an *engineering shear strain* \gamma
    definition and adds a compensating normal strain so that the overall
    deformation is volume-conserving along this specific strain path.

    For sufficiently small \gamma the two become essentially
    equivalent, but they differ for larger strains. Fractional
    coordinates are preserved, Cartesian coordinates are rebuilt from
    the deformed cell, and copied energy/force labels are cleared.

    Parameters
    ----------
    direction : int
        Shear plane (1=xy, 2=xz, 3=yz)
    gamma_min, gamma_max : float
        Range of engineering shear strain \gamma
    steps : int
        Number of steps

    Notes
    -----
    The ``gamma_min`` and ``gamma_max`` parameters specify the engineering
    shear strain γ_xy. Note the formula for the compensating normal strain:
    ε_zz = γ²/(4-γ²).

    - The formulation has a singularity at :math:`|\gamma| = 2`, where
      :math:`\epsilon_{zz}` diverges. In practice, require
      :math:`|\gamma_{\min}| < 2` and :math:`|\gamma_{\max}| < 2`.
    - For structure-space sampling, a typical range is -0.4 to +0.4. Staying
      well below :math:`|\gamma| = 1.0` avoids extreme cell deformations, and
      values above :math:`|\gamma| > 0.6` may produce highly skewed or
      unstable structures.
    - For elastic constant calculations targeting ``C44``, stay in the linear
      regime around -0.1 to +0.1, use at least 5-7 points with both positive
      and negative shear values, and ensure :math:`|\gamma| \ll 1`. In this
      limit the result approaches the simple-shear formulation.
    - For :math:`|\gamma| < 0.1`,
      MonoclinicStrainTransformation and ShearStrainTransformation yield
      nearly identical results. They diverge at larger strains because they
      enforce volume conservation differently.
    """

    def __init__(self, direction: int, gamma_min: float,
                 gamma_max: float, steps: int):
        if direction not in (1, 2, 3):
            raise ValueError(f"direction must be 1, 2, or 3, got {direction}")
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        if gamma_min >= gamma_max:
            raise ValueError("gamma_min must be less than gamma_max")

        self.direction = direction
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.steps = steps

    def apply_transformation(
        self,
        structure: AtomicStructure,
        **kwargs,
    ) -> Iterator[AtomicStructure]:
        """Apply monoclinic strain pattern to structure.

        Yields structures with volume-conserving monoclinic strain.
        """
        if not structure.pbc:
            raise ValueError(
                "MonoclinicStrainTransformation requires periodic structure"
            )

        def _gen() -> Iterator[AtomicStructure]:
            gamma_values = np.linspace(
                self.gamma_min, self.gamma_max, self.steps)

            for gamma in gamma_values:
                avec0 = structure.avec[-1]

                # Map general direction to permutations so that utils helper
                # (which is xy by definition) can be reused.
                perm = {
                    1: (0, 1, 2),  # xy
                    2: (0, 2, 1),  # xz -> map to xy by swapping y<->z
                    3: (1, 2, 0),  # yz -> map to xy by cycling
                }[self.direction]

                A_perm = avec0[list(perm), :]
                A_strained = geom_utils.strain_monoclinic(A_perm, gamma)

                invperm = np.argsort(np.array(perm))
                A_back = A_strained[invperm, :]

                strained = structure.copy()
                strained.update_cell(A_back, preserve="fractional")
                yield strained

        return _gen()
