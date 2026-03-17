"""
Tests for the transformations infrastructure.

Tests the base transformation framework and chain execution
with iterator-based API.
"""

import itertools

import numpy as np
import pytest

from aenet.geometry import AtomicStructure
from aenet.geometry.transformations import (
    TransformationABC,
    TransformationChain,
)


# Test helper: Simple transformations for testing
class IdentityTransformation(TransformationABC):
    """Transformation that yields the input unchanged."""

    def apply_transformation(self, structure, **kwargs):
        """Yield structure unchanged."""
        yield structure


class DuplicateTransformation(TransformationABC):
    """Transformation that yields N copies of input."""

    def __init__(self, n_copies=2):
        """Initialize with number of copies."""
        self.n_copies = n_copies

    def apply_transformation(self, structure, **kwargs):
        """Yield N copies of the structure."""
        for _ in range(self.n_copies):
            yield structure.copy()


class DisplaceAtomsTransformation(TransformationABC):
    """Simple transformation that displaces all atoms."""

    def __init__(self, displacement=0.1):
        """Initialize with displacement magnitude."""
        self.displacement = displacement

    def apply_transformation(self, structure, **kwargs):
        """Displace all atoms by constant amount in x direction."""
        result = structure.copy()
        for i in range(result.nframes):
            result.coords[i][:, 0] += self.displacement
        yield result


# Fixtures
@pytest.fixture
def simple_structure():
    """Create a simple test structure."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    types = ['H', 'H', 'O']
    avec = np.array([
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 3.0],
    ])
    return AtomicStructure(coords, types, avec=avec)


# Test TransformationABC
def test_transformation_abc_is_abstract():
    """Test that TransformationABC requires implementation."""
    class UnimplementedTransformation(TransformationABC):
        pass

    # Instantiation should fail for abstract class
    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class"
    ):
        UnimplementedTransformation()


def test_transformation_abc_requires_method():
    """Test that apply_transformation must be implemented."""
    # This should fail to instantiate
    with pytest.raises(TypeError):
        class BadTransformation(TransformationABC):
            pass
        BadTransformation()


# Test basic transformation
def test_identity_transformation_yields_structure(simple_structure):
    """Test that transformation yields structures."""
    transform = IdentityTransformation()
    result_iter = transform.apply_transformation(simple_structure)

    # Should be an iterator
    assert hasattr(result_iter, '__iter__')
    assert hasattr(result_iter, '__next__')

    # Consume it
    results = list(result_iter)
    assert len(results) == 1
    assert isinstance(results[0], AtomicStructure)
    assert results[0].natoms == simple_structure.natoms


def test_duplicate_transformation_yields_multiple(simple_structure):
    """Test transformation that yields multiple structures."""
    transform = DuplicateTransformation(n_copies=5)
    results = list(transform.apply_transformation(simple_structure))

    assert len(results) == 5
    assert all(isinstance(s, AtomicStructure) for s in results)


def test_transformation_lazy_evaluation(simple_structure):
    """Test that transformations use lazy evaluation."""
    call_count = [0]

    class CountingTransformation(TransformationABC):
        def apply_transformation(self, structure, **kwargs):
            for i in range(10):
                call_count[0] += 1
                yield structure.copy()

    transform = CountingTransformation()
    iterator = transform.apply_transformation(simple_structure)

    # Iterator created but not executed
    assert call_count[0] == 0

    # Take first 3 items
    first_three = list(itertools.islice(iterator, 3))
    assert len(first_three) == 3
    # Only 3 iterations should have run
    assert call_count[0] == 3


def test_transformation_partial_consumption(simple_structure):
    """Test partial consumption of transformation iterator."""
    transform = DuplicateTransformation(n_copies=100)
    iterator = transform.apply_transformation(simple_structure)

    # Take only first 5
    first_five = list(itertools.islice(iterator, 5))
    assert len(first_five) == 5


def test_transformation_with_kwargs(simple_structure):
    """Test that transformations accept **kwargs."""
    class KwargsTransformation(TransformationABC):
        def apply_transformation(self, structure, **kwargs):
            # kwargs should be passed through
            assert 'test_param' in kwargs
            assert kwargs['test_param'] == 42
            yield structure

    transform = KwargsTransformation()
    results = list(transform.apply_transformation(
        simple_structure, test_param=42
    ))
    assert len(results) == 1


# Test TransformationChain
def test_chain_initialization():
    """Test chain initialization."""
    chain = TransformationChain([IdentityTransformation()])
    assert len(chain.transformations) == 1


def test_chain_empty_transformations(simple_structure):
    """Test chain with no transformations."""
    chain = TransformationChain([])
    results = list(chain.apply_transformation(simple_structure))

    # Should return input unchanged
    assert len(results) == 1
    assert results[0] == simple_structure


def test_chain_single_transformation(simple_structure):
    """Test chain with single transformation."""
    chain = TransformationChain([IdentityTransformation()])
    results = list(chain.apply_transformation(simple_structure))

    assert len(results) == 1
    assert isinstance(results[0], AtomicStructure)


def test_chain_multiple_transformations(simple_structure):
    """Test chain with multiple transformations."""
    chain = TransformationChain([
        DisplaceAtomsTransformation(0.1),
        DisplaceAtomsTransformation(0.1),
    ])
    results = list(chain.apply_transformation(simple_structure))

    assert len(results) == 1
    # Total displacement should be 0.2
    expected_x = simple_structure.coords[0][0, 0] + 0.2
    assert np.isclose(results[0].coords[0][0, 0], expected_x)


def test_chain_one_to_many(simple_structure):
    """Test chain with one-to-many transformation."""
    chain = TransformationChain([
        DuplicateTransformation(n_copies=3),
    ])
    results = list(chain.apply_transformation(simple_structure))

    # Should generate 3 structures
    assert len(results) == 3


def test_chain_combinatorial_growth(simple_structure):
    """Test combinatorial growth in chain."""
    chain = TransformationChain([
        DuplicateTransformation(n_copies=3),
        DuplicateTransformation(n_copies=2),
    ])
    results = list(chain.apply_transformation(simple_structure))

    # 3 * 2 = 6 structures
    assert len(results) == 6


def test_chain_lazy_evaluation(simple_structure):
    """Test that chain uses lazy evaluation."""
    chain = TransformationChain([
        DuplicateTransformation(n_copies=100),
        DuplicateTransformation(n_copies=100),
    ])

    iterator = chain.apply_transformation(simple_structure)

    # Take only first 10 (would be 10000 without lazy evaluation)
    first_ten = list(itertools.islice(iterator, 10))
    assert len(first_ten) == 10


def test_chain_partial_consumption(simple_structure):
    """Test partial consumption with chain."""
    chain = TransformationChain([
        DuplicateTransformation(n_copies=10),
    ])

    iterator = chain.apply_transformation(simple_structure)

    # Consume only 5
    first_five = list(itertools.islice(iterator, 5))
    assert len(first_five) == 5


def test_chain_with_kwargs(simple_structure):
    """Test that chain passes kwargs to transformations."""
    class KwargsTransformation(TransformationABC):
        def apply_transformation(self, structure, **kwargs):
            assert 'test_param' in kwargs
            yield structure

    chain = TransformationChain([KwargsTransformation()])
    results = list(chain.apply_transformation(
        simple_structure, test_param=42
    ))
    assert len(results) == 1


# Test with multi-frame structures
def test_transformation_with_multiframe_structure(simple_structure):
    """Test transformation with multi-frame structure."""
    # Add a second frame
    simple_structure.add_frame(
        simple_structure.coords[0] + 0.1,
        avec=simple_structure.avec[0]
    )

    transform = DisplaceAtomsTransformation(0.5)
    results = list(transform.apply_transformation(simple_structure))

    # Both frames should be displaced
    assert len(results) == 1
    result = results[0]
    assert result.nframes == 2
    assert np.allclose(
        result.coords[0][0, 0],
        simple_structure.coords[0][0, 0] + 0.5
    )
    assert np.allclose(
        result.coords[1][0, 0],
        simple_structure.coords[1][0, 0] + 0.5
    )


# Test iterator behavior
def test_transformation_iterator_can_be_reused(simple_structure):
    """Test that transformation can be called multiple times."""
    transform = DuplicateTransformation(n_copies=3)

    # First call
    results1 = list(transform.apply_transformation(simple_structure))
    assert len(results1) == 3

    # Second call should work independently
    results2 = list(transform.apply_transformation(simple_structure))
    assert len(results2) == 3


def test_chain_iterator_can_be_reused(simple_structure):
    """Test that chain can be called multiple times."""
    chain = TransformationChain([
        DuplicateTransformation(n_copies=2),
    ])

    # First call
    results1 = list(chain.apply_transformation(simple_structure))
    assert len(results1) == 2

    # Second call
    results2 = list(chain.apply_transformation(simple_structure))
    assert len(results2) == 2


def test_transformation_empty_output(simple_structure):
    """Test transformation that yields no structures."""
    class EmptyTransformation(TransformationABC):
        def apply_transformation(self, structure, **kwargs):
            # Yield nothing
            return
            yield  # Make it a generator

    transform = EmptyTransformation()
    results = list(transform.apply_transformation(simple_structure))
    assert len(results) == 0


def test_chain_with_empty_intermediate(simple_structure):
    """Test chain where intermediate transformation yields nothing."""
    class EmptyTransformation(TransformationABC):
        def apply_transformation(self, structure, **kwargs):
            return
            yield

    chain = TransformationChain([
        DuplicateTransformation(n_copies=3),
        EmptyTransformation(),
        DuplicateTransformation(n_copies=2),
    ])

    results = list(chain.apply_transformation(simple_structure))
    # Empty intermediate means no output
    assert len(results) == 0


# Integration tests
def test_full_chain_workflow(simple_structure):
    """Test complete chain workflow."""
    chain = TransformationChain([
        DuplicateTransformation(n_copies=3),
        DisplaceAtomsTransformation(0.1),
    ])

    results = list(chain.apply_transformation(simple_structure))

    # Should have 3 structures (3 from first, 1 from each in second)
    assert len(results) == 3

    # All should be displaced
    for result in results:
        expected_x = simple_structure.coords[0][0, 0] + 0.1
        assert np.isclose(result.coords[0][0, 0], expected_x)


def test_manual_filtering_with_itertools(simple_structure):
    """Test that users can filter results with itertools."""
    class NumberedTransformation(TransformationABC):
        def __init__(self, n_copies):
            self.n_copies = n_copies

        def apply_transformation(self, structure, **kwargs):
            for i in range(self.n_copies):
                s = structure.copy()
                s.coords[0][0, 0] += i * 0.1
                yield s

    transform = NumberedTransformation(n_copies=10)
    iterator = transform.apply_transformation(simple_structure)

    # User can filter however they want
    filtered = [s for s in iterator if s.coords[0][0, 0] < 0.5]
    assert len(filtered) < 10


def test_manual_ranking_with_sorted(simple_structure):
    """Test that users can rank results with sorted()."""
    class NumberedTransformation(TransformationABC):
        def __init__(self, n_copies):
            self.n_copies = n_copies

        def apply_transformation(self, structure, **kwargs):
            for i in range(self.n_copies):
                s = structure.copy()
                s.coords[0][0, 0] += i * 0.1
                yield s

    transform = NumberedTransformation(n_copies=5)
    results = list(transform.apply_transformation(simple_structure))

    # Sort by x-coordinate (descending)
    sorted_results = sorted(
        results,
        key=lambda s: s.coords[0][0, 0],
        reverse=True
    )

    # Largest should be first
    assert (sorted_results[0].coords[0][0, 0] >
            sorted_results[-1].coords[0][0, 0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
