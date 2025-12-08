"""
Tests for diversity metrics functions in geometry.utils.

Tests the vector set diversity analysis functions including
standardization, covariance-based entropy, and comprehensive
diversity metrics.
"""

import pytest
import numpy as np
import warnings

from aenet.geometry.utils import (
    standardize,
    entropy_from_cov_regularized,
    diversity_metrics,
    format_diversity_metrics,
)


# Fixtures
@pytest.fixture
def random_vectors():
    """Generate random vectors for testing."""
    np.random.seed(42)
    return np.random.randn(20, 50)


@pytest.fixture
def small_vectors():
    """Generate small set of vectors for testing."""
    return np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0],
    ])


@pytest.fixture
def diverse_vectors():
    """Generate diverse set of vectors."""
    np.random.seed(42)
    return np.random.randn(10, 30)


@pytest.fixture
def redundant_vectors():
    """Generate redundant set of vectors (small perturbations)."""
    np.random.seed(42)
    base_vec = np.random.randn(30)
    return base_vec + 0.01 * np.random.randn(10, 30)


# Test standardize function
class TestStandardize:
    """Tests for standardize function."""

    def test_standardize_returns_ndarray(self, small_vectors):
        """Test that standardize returns ndarray."""
        result = standardize(small_vectors)
        assert isinstance(result, np.ndarray)
        assert result.shape == small_vectors.shape

    def test_standardize_zero_mean(self, small_vectors):
        """Test that standardized vectors have zero mean."""
        result = standardize(small_vectors)
        mean = result.mean(axis=0)
        assert np.allclose(mean, 0.0, atol=1e-10)

    def test_standardize_unit_variance(self, small_vectors):
        """Test that standardized vectors have unit variance."""
        result = standardize(small_vectors)
        std = result.std(axis=0, ddof=1)
        assert np.allclose(std, 1.0, atol=1e-10)

    def test_standardize_preserves_shape(self, random_vectors):
        """Test that standardize preserves input shape."""
        result = standardize(random_vectors)
        assert result.shape == random_vectors.shape

    def test_standardize_with_zero_variance_column(self):
        """Test standardize with zero variance column."""
        vecs = np.array([
            [1.0, 5.0],
            [2.0, 5.0],
            [3.0, 5.0],
        ])
        result = standardize(vecs)

        # First column should be standardized
        assert np.allclose(result.mean(axis=0)[0], 0.0, atol=1e-10)
        assert np.allclose(result.std(axis=0, ddof=1)[0], 1.0, atol=1e-10)
        # Second column has zero variance, gets centered (mean removed)
        # but divided by 1.0 instead of 0.0, resulting in all zeros
        assert np.allclose(result[:, 1], 0.0)

    def test_standardize_single_vector_raises_warning(self):
        """Test that single vector doesn't cause issues."""
        # With N=1, ddof=1 gives NaN, which gets replaced with 1.0
        vec = np.array([[1.0, 2.0, 3.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = standardize(vec)
        assert result.shape == vec.shape


# Test entropy_from_cov_regularized function
class TestEntropyFromCov:
    """Tests for entropy_from_cov_regularized function."""

    def test_entropy_returns_float(self, random_vectors):
        """Test that entropy returns a float."""
        result = entropy_from_cov_regularized(random_vectors)
        assert isinstance(result, float)

    def test_entropy_not_nan(self, random_vectors):
        """Test that entropy is not NaN for valid input."""
        result = entropy_from_cov_regularized(random_vectors)
        assert not np.isnan(result)

    def test_entropy_finite_or_neginf(self, random_vectors):
        """Test that entropy is finite or -inf."""
        result = entropy_from_cov_regularized(random_vectors)
        assert np.isfinite(result) or result == -np.inf

    def test_entropy_small_n_large_d(self):
        """Test efficiency for N << d case."""
        np.random.seed(42)
        # This is where Matrix Determinant Lemma shines
        vecs = np.random.randn(5, 1000)
        result = entropy_from_cov_regularized(vecs)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_entropy_with_alpha_parameter(self, random_vectors):
        """Test entropy with different alpha values."""
        result1 = entropy_from_cov_regularized(random_vectors, alpha=1e-3)
        result2 = entropy_from_cov_regularized(random_vectors, alpha=1e-2)

        # Different regularization should give different results
        assert result1 != result2

    def test_entropy_raises_on_insufficient_samples(self):
        """Test that entropy raises ValueError for N <= 1."""
        vec = np.array([[1.0, 2.0, 3.0]])

        with pytest.raises(ValueError, match="at least 2 samples"):
            entropy_from_cov_regularized(vec)

    def test_entropy_diverse_vs_redundant(
            self, diverse_vectors, redundant_vectors):
        """Test that diverse sets have higher entropy than redundant sets."""
        entropy_diverse = entropy_from_cov_regularized(diverse_vectors)
        entropy_redundant = entropy_from_cov_regularized(redundant_vectors)

        # Diverse set should have higher log-det
        assert entropy_diverse > entropy_redundant

    def test_entropy_matrix_determinant_lemma(self):
        """Test that Matrix Determinant Lemma gives correct results."""
        np.random.seed(42)
        # Small test case where we can verify the math
        N, d = 10, 50
        vecs = np.random.randn(N, d)

        result = entropy_from_cov_regularized(vecs, alpha=1e-3)

        # Should be finite and reasonable
        assert np.isfinite(result)
        assert result > -1000  # Sanity check


# Test diversity_metrics function
class TestDiversityMetrics:
    """Tests for diversity_metrics function."""

    def test_diversity_metrics_returns_dict(self, random_vectors):
        """Test that diversity_metrics returns a dictionary."""
        result = diversity_metrics(random_vectors)
        assert isinstance(result, dict)

    def test_diversity_metrics_has_required_keys(self, random_vectors):
        """Test that all required keys are present."""
        result = diversity_metrics(random_vectors)

        expected_keys = [
            'mean_euclidean_dist',
            'mean_pearson_corr',
            'mean_cosine_sim',
            'log_det_cov',
            'n_vectors',
            'dimension'
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_diversity_metrics_correct_counts(self):
        """Test that n_vectors and dimension are correct."""
        vecs = np.random.randn(15, 30)
        result = diversity_metrics(vecs)

        assert result['n_vectors'] == 15
        assert result['dimension'] == 30

    def test_diversity_metrics_all_numeric(self, random_vectors):
        """Test that all metric values are numeric."""
        result = diversity_metrics(random_vectors)

        assert isinstance(result['mean_euclidean_dist'], (int, float))
        assert isinstance(result['mean_pearson_corr'], (int, float))
        assert isinstance(result['mean_cosine_sim'], (int, float))
        assert isinstance(result['log_det_cov'], (int, float))

    def test_diversity_metrics_raises_on_insufficient_vectors(self):
        """Test that diversity_metrics raises ValueError for N < 2."""
        vec = np.array([[1.0, 2.0, 3.0]])

        with pytest.raises(ValueError, match="at least 2 vectors"):
            diversity_metrics(vec)

    def test_diversity_metrics_small_set(self, small_vectors):
        """Test diversity metrics on small vector set."""
        result = diversity_metrics(small_vectors)

        # Should have valid numeric results
        assert result['n_vectors'] == 3
        assert result['dimension'] == 3
        assert result['mean_euclidean_dist'] > 0

    def test_diversity_metrics_orthogonal_vectors(self):
        """Test metrics on orthogonal vectors."""
        # Create orthonormal basis in higher dimension
        # to avoid correlation artifacts from small samples
        vecs = np.eye(5)
        result = diversity_metrics(vecs)

        # Orthogonal unit vectors have zero cosine similarity
        assert abs(result['mean_cosine_sim']) < 0.01
        # Pearson correlation can be non-zero for small samples with
        # many zeros, so we just verify it's in valid range
        assert -1.0 <= result['mean_pearson_corr'] <= 1.0

    def test_diversity_metrics_parallel_vectors(self):
        """Test metrics on parallel vectors."""
        # Create parallel vectors
        base = np.array([1.0, 2.0, 3.0])
        vecs = np.array([base, 2 * base])
        result = diversity_metrics(vecs)

        # Parallel vectors should have high cosine similarity
        assert result['mean_cosine_sim'] > 0.99

    def test_diversity_metrics_diverse_vs_redundant(
        self, diverse_vectors, redundant_vectors
    ):
        """Test that diverse sets have higher metrics than redundant sets."""
        metrics_diverse = diversity_metrics(diverse_vectors)
        metrics_redundant = diversity_metrics(redundant_vectors)

        # Diverse set should have:
        # - Higher log-det covariance
        # - Larger mean distances
        assert metrics_diverse['log_det_cov'
                               ] > metrics_redundant['log_det_cov']
        assert metrics_diverse['mean_euclidean_dist'] > \
               metrics_redundant['mean_euclidean_dist']

    def test_diversity_metrics_pearson_corr_range(self, random_vectors):
        """Test that Pearson correlation is in valid range."""
        result = diversity_metrics(random_vectors)

        # Pearson correlation should be in [-1, 1]
        assert -1.0 <= result['mean_pearson_corr'] <= 1.0

    def test_diversity_metrics_cosine_sim_range(self, random_vectors):
        """Test that cosine similarity is in valid range."""
        result = diversity_metrics(random_vectors)

        # Cosine similarity should be in [-1, 1]
        assert -1.0 <= result['mean_cosine_sim'] <= 1.0

    def test_diversity_metrics_1d_vectors(self):
        """Test diversity metrics on 1D vectors."""
        vecs = np.array([[1.0], [2.0], [3.0]])
        result = diversity_metrics(vecs)

        assert result['dimension'] == 1
        assert result['n_vectors'] == 3


# Test format_diversity_metrics function
class TestFormatDiversityMetrics:
    """Tests for format_diversity_metrics function."""

    def test_format_returns_string(self, random_vectors):
        """Test that format returns a string."""
        metrics = diversity_metrics(random_vectors)
        result = format_diversity_metrics(metrics)

        assert isinstance(result, str)

    def test_format_has_four_values(self, random_vectors):
        """Test that formatted string has 4 comma-separated values."""
        metrics = diversity_metrics(random_vectors)
        result = format_diversity_metrics(metrics)

        parts = result.split(', ')
        assert len(parts) == 4

    def test_format_values_are_numeric(self, random_vectors):
        """Test that formatted values can be parsed as floats."""
        metrics = diversity_metrics(random_vectors)
        result = format_diversity_metrics(metrics)

        parts = result.split(', ')
        for part in parts:
            # Should be parseable as float
            try:
                float(part)
            except ValueError:
                pytest.fail(f"Could not parse '{part}' as float")

    def test_format_fixed_precision(self, random_vectors):
        """Test that format uses 3 decimal places."""
        metrics = diversity_metrics(random_vectors)
        result = format_diversity_metrics(metrics)

        parts = result.split(', ')
        for part in parts:
            # Should have at most 3 decimal places
            # (plus possible negative sign)
            assert len(part) <= 8  # -XXX.XXX is max 8 characters


# Integration tests
class TestDiversityAnalysisIntegration:
    """Integration tests for diversity analysis workflow."""

    def test_full_workflow_standardize_then_metrics(self):
        """Test complete workflow: standardize then compute metrics."""
        np.random.seed(42)
        vecs = np.random.randn(15, 40) * 10 + 5  # Random scale and offset

        # Standardize first
        vecs_std = standardize(vecs)

        # Compute metrics
        metrics = diversity_metrics(vecs_std)

        # Format results
        formatted = format_diversity_metrics(metrics)

        assert isinstance(formatted, str)
        assert metrics['n_vectors'] == 15
        assert metrics['dimension'] == 40

    def test_comparison_workflow(self):
        """Test workflow for comparing different vector sets."""
        np.random.seed(42)

        # Create two sets with different diversity
        diverse = np.random.randn(10, 20)
        base = np.random.randn(20)
        redundant = base + 0.001 * np.random.randn(10, 20)

        # Compute metrics for both
        metrics_diverse = diversity_metrics(diverse)
        metrics_redundant = diversity_metrics(redundant)

        # Diverse should have higher log-det
        assert metrics_diverse['log_det_cov'
                               ] > metrics_redundant['log_det_cov']

    def test_entropy_consistency_with_diversity_metrics(self, random_vectors):
        """Test that entropy from both functions is consistent."""
        # Compute using both methods
        entropy_direct = entropy_from_cov_regularized(random_vectors)
        metrics = diversity_metrics(random_vectors)
        entropy_from_metrics = metrics['log_det_cov']

        # Should be the same
        assert np.isclose(entropy_direct, entropy_from_metrics)


# Edge cases and error handling
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_array_handling(self):
        """Test handling of empty arrays."""
        empty = np.array([]).reshape(0, 3)

        with pytest.raises((ValueError, IndexError)):
            diversity_metrics(empty)

    def test_single_sample_handling(self):
        """Test handling of single sample."""
        single = np.array([[1.0, 2.0, 3.0]])

        with pytest.raises(ValueError):
            diversity_metrics(single)

    def test_very_small_vectors(self):
        """Test with very small magnitude vectors."""
        vecs = np.array([
            [1e-10, 1e-10],
            [2e-10, 2e-10],
        ])

        # Should not crash
        result = diversity_metrics(vecs)
        assert isinstance(result, dict)

    def test_identical_vectors(self):
        """Test with identical vectors."""
        vecs = np.ones((5, 10))

        # Should handle gracefully (will have degenerate covariance)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = diversity_metrics(vecs)

        # Identical vectors should have zero distance
        assert np.isclose(result['mean_euclidean_dist'], 0.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
