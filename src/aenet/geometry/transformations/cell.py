"""Cell (lattice) transformations.

This module contains transformations that primarily modify the unit cell
(lattice vectors) of a periodic
:class:`~aenet.geometry.structure.AtomicStructure`.

Notes
-----
All transformations in this module require periodic boundary conditions
(``structure.pbc`` must be True).
"""

import logging
from collections.abc import Iterator
from typing import Optional

import numpy as np

from .. import utils as geom_utils
from ..structure import AtomicStructure
from .base import TransformationABC

logger = logging.getLogger(__name__)


class CellVolumeTransformation(TransformationABC):
    """Uniformly scale unit cell vectors.

    This transformation generates structures with different volumes by
    uniformly scaling the lattice vectors. Fractional coordinates remain
    unchanged, Cartesian coordinates are recomputed from the updated
    cell, and copied energy/force labels are cleared because they are
    no longer valid for the deformed geometry.

    Physical/engineering meaning
    ----------------------------
    This is a hydrostatic (isotropic) scaling of the lattice. In a
    thermodynamic context it resembles sampling different volumes (e.g.,
    equation of state fits), but without relaxing internal degrees of
    freedom.  It can be used, for example, to calculate the bulk modulus.

    Parameters
    ----------
    min_percent : float, optional
        Minimum percentage change from original lattice scaling
        (default: -5.0)
    max_percent : float, optional
        Maximum percentage change from original lattice scaling
        (default: 5.0)
    steps : int, optional
        Number of scaling steps (default: 5)
    """

    def __init__(
        self,
        min_percent: float = -5.0,
        max_percent: float = 5.0,
        steps: int = 5,
    ):
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        if min_percent >= max_percent:
            raise ValueError(
                f"min_percent ({min_percent}) must be less than "
                f"max_percent ({max_percent})"
            )
        self.min_percent = min_percent
        self.max_percent = max_percent
        self.steps = steps

    def apply_transformation(
        self,
        structure: AtomicStructure,
        **kwargs,
    ) -> Iterator[AtomicStructure]:
        """Apply uniform volume scaling to structure.

        Yields structures with different cell volumes while preserving
        fractional coordinates.
        """
        if not structure.pbc:
            raise ValueError(
                "CellVolumeTransformation requires periodic structure"
            )

        def _gen() -> Iterator[AtomicStructure]:
            min_scale = 1.0 + self.min_percent / 100.0
            max_scale = 1.0 + self.max_percent / 100.0
            scale_factors = np.linspace(min_scale, max_scale, self.steps)

            for scale in scale_factors:
                scaled = structure.copy()
                scaled.update_cell(
                    structure.avec[-1] * scale,
                    preserve="fractional",
                )
                yield scaled

        return _gen()


class CellTransformationMatrix(TransformationABC):
    """General integer cell/basis transformation using a 3x3 matrix.

    This wraps :func:`aenet.geometry.utils.transform_cell`.

    The transformation matrix ``T`` is applied as::

        A' = T · A

    where rows of ``A`` are the lattice vectors. Atomic coordinates are
    transformed accordingly and redundant periodic images are created
    (or removed) so that the atom count is consistent with the volume
    scaling factor.

    Purpose
    -------
    This is a *change of lattice basis* that does not actually alter
    the atomic structure, but rather how it is represented. This is
    useful for generating:

    - different supercell shapes for the same primitive cell
    - commensurate cells for defects, phonons, or disordered sampling
    - as a first step when constructing surface slabs from bulk structures

    Parameters
    ----------
    T : array_like shape (3, 3)
        Transformation matrix. In practice this should be integer-valued
        for supercell / basis transformations.
    sort : int or None, optional
        Sorting behavior passed to ``utils.transform_cell`` (default: 2).
        ``2`` sorts by fractional z, then by type. ``None`` disables the
        coordinate-based sort (type sort still happens).

    Notes
    -----
    This currently relies on utilities that expect *fractional*
    coordinates. We convert AtomicStructure Cartesian coordinates to
    fractional, apply the transformation, then convert back to Cartesian.
    """

    def __init__(self, T, sort: Optional[int] = 2):
        self.T = np.array(T, dtype=float)
        if self.T.shape != (3, 3):
            raise ValueError(f"T must be shape (3,3), got {self.T.shape}")
        self.sort = sort

    def apply_transformation(
        self,
        structure: AtomicStructure,
        **kwargs,
    ) -> Iterator[AtomicStructure]:
        """Apply cell basis transformation to structure.

        Yields a single transformed structure with new cell and coordinates.
        """
        if not structure.pbc:
            raise ValueError(
                "CellTransformationMatrix requires a periodic structure"
            )

        # AtomicStructure stores Cartesian coordinates; utils.transform_cell
        # expects fractional coordinates.
        avec = structure.avec[-1]
        frac = structure.coords[-1] @ structure.bvec[-1]

        # utils.transform_cell expects types in sortable form; AtomicStructure
        # stores .types as list/ndarray.
        types = np.asarray(structure.types)

        avec_T, coords_frac_T, types_T = geom_utils.transform_cell(
            avec=avec,
            coords=frac,
            types=types,
            T=self.T,
            sort=self.sort,
        )

        transformed = structure.copy()
        transformed.avec[-1] = np.array(avec_T, dtype=float)
        transformed.bvec[-1] = np.linalg.inv(transformed.avec[-1])
        transformed.types = list(types_T)
        transformed.coords[-1] = np.array(coords_frac_T) @ transformed.avec[-1]

        def _gen() -> Iterator[AtomicStructure]:
            yield transformed

        return _gen()
