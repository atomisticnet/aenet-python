"""
Base classes for structure transformations.

This module provides the abstract base class and chain functionality
for all structure transformations in aenet.
"""

import abc
import logging
from collections.abc import Iterator

from ..structure import AtomicStructure

__author__ = "Alexander Urban, Nongnuch Artrith"
__date__ = "2025-12-01"

# Set up logging
logger = logging.getLogger(__name__)


class TransformationABC(abc.ABC):
    """
    Abstract base class for all structure transformations.

    All transformations must implement apply_transformation which yields
    transformed AtomicStructure objects.
    """

    @abc.abstractmethod
    def apply_transformation(
        self,
        structure: AtomicStructure,
        **kwargs
    ) -> Iterator[AtomicStructure]:
        """
        Apply transformation to structure.

        Parameters
        ----------
        structure : AtomicStructure
            Input structure to transform
        **kwargs
            Additional keyword arguments for forward compatibility

        Yields
        ------
        AtomicStructure
            Transformed structures
        """
        raise NotImplementedError


class TransformationChain:
    """
    Chain multiple transformations with lazy evaluation.

    This class allows sequential application of multiple transformations
    using itertools for memory-efficient streaming.

    Parameters
    ----------
    transformations : list of TransformationABC
        List of transformations to apply sequentially

    Examples
    --------
    >>> chain = TransformationChain([transform1, transform2])
    >>> for structure in chain.apply_transformation(input_structure):
    ...     process(structure)

    >>> # Or consume all at once
    >>> structures = list(chain.apply_transformation(input_structure))
    """

    def __init__(self, transformations: list[TransformationABC]):
        """Initialize transformation chain."""
        self.transformations = transformations
        logger.info(
            f"TransformationChain initialized with {len(transformations)} "
            f"transformations"
        )

    def apply_transformation(
        self,
        structure: AtomicStructure,
        **kwargs
    ) -> Iterator[AtomicStructure]:
        """
        Apply all transformations sequentially using depth-first streaming.

        This avoids iterator aliasing issues and guarantees correct
        combinatorics (including zero-output steps).

        Parameters
        ----------
        structure : AtomicStructure
            Input structure to transform
        **kwargs
            Additional keyword arguments passed to transformations

        Returns
        -------
        Iterator[AtomicStructure]
            Transformed structures from the chain
        """
        logger.info("Starting chain with 1 input structure")

        n_steps = len(self.transformations)

        def _apply_from(step_idx: int,
                        s: AtomicStructure) -> Iterator[AtomicStructure]:
            # If no more transforms, yield the structure
            if step_idx >= n_steps:
                yield s
                return

            transform = self.transformations[step_idx]
            logger.debug(
                f"Chain step {step_idx+1}/{n_steps}: "
                f"{transform.__class__.__name__}"
            )

            # For each output of this transform, continue with remaining steps
            for out in transform.apply_transformation(s, **kwargs):
                yield from _apply_from(step_idx + 1, out)

        # One input structure to start with
        # Return a generator that streams depth-first results
        return _apply_from(0, structure)
