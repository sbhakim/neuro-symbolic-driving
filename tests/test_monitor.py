# tests/test_monitor.py

import pytest

from src.stl_monitor import robustness, compute_rho_min, compute_satisfaction_rate


class TestRobustness:
    """Unit tests for robustness computation."""

    def test_positive_robustness(self):
        """Test robustness when distance > d_min (safe margin)."""
        # arrange
        distance = 25.0
        d_min = 10.0

        # act
        rho = robustness(distance, d_min)

        # assert
        assert rho == pytest.approx(15.0)

    def test_zero_robustness(self):
        """Test robustness when distance = d_min (exactly at threshold)."""
        # arrange
        distance = 10.0
        d_min = 10.0

        # act
        rho = robustness(distance, d_min)

        # assert
        assert rho == pytest.approx(0.0)

    def test_negative_robustness(self):
        """Test robustness when distance < d_min (violation)."""
        # arrange
        distance = 5.0
        d_min = 10.0

        # act
        rho = robustness(distance, d_min)

        # assert
        assert rho == pytest.approx(-5.0)


class TestComputeRhoMin:
    """Unit tests for minimum robustness computation."""

    def test_single_distance(self):
        """Test rho_min with single distance value."""
        # arrange
        distances = [20.0]
        d_min = 10.0

        # act
        rho_min = compute_rho_min(distances, d_min)

        # assert
        assert rho_min == pytest.approx(10.0)

    def test_all_positive_robustness(self):
        """Test rho_min when all distances are safe."""
        # arrange
        distances = [25.0, 30.0, 20.0, 28.0]
        d_min = 10.0

        # act
        rho_min = compute_rho_min(distances, d_min)

        # assert - minimum is 20.0 - 10.0 = 10.0
        assert rho_min == pytest.approx(10.0)

    def test_mixed_robustness(self):
        """Test rho_min with mix of positive and negative robustness."""
        # arrange
        distances = [25.0, 8.0, 15.0, 12.0]
        d_min = 10.0

        # act
        rho_min = compute_rho_min(distances, d_min)

        # assert - minimum is 8.0 - 10.0 = -2.0
        assert rho_min == pytest.approx(-2.0)

    def test_all_negative_robustness(self):
        """Test rho_min when all distances violate safety."""
        # arrange
        distances = [5.0, 3.0, 7.0, 2.0]
        d_min = 10.0

        # act
        rho_min = compute_rho_min(distances, d_min)

        # assert - minimum is 2.0 - 10.0 = -8.0
        assert rho_min == pytest.approx(-8.0)


class TestComputeSatisfactionRate:
    """Unit tests for satisfaction rate computation."""

    def test_empty_list(self):
        """Test satisfaction rate returns 0.0 for empty list."""
        # arrange
        distances = []
        d_min = 10.0

        # act
        sr = compute_satisfaction_rate(distances, d_min)

        # assert
        assert sr == pytest.approx(0.0)

    def test_all_satisfied(self):
        """Test satisfaction rate when all distances are safe (rho > 0)."""
        # arrange
        distances = [15.0, 20.0, 25.0, 18.0]
        d_min = 10.0

        # act
        sr = compute_satisfaction_rate(distances, d_min)

        # assert - all have rho > 0, so SR = 1.0
        assert sr == pytest.approx(1.0)

    def test_none_satisfied(self):
        """Test satisfaction rate when no distances are safe (all rho <= 0)."""
        # arrange
        distances = [5.0, 8.0, 3.0, 10.0]  # note: 10.0 gives rho=0, not > 0
        d_min = 10.0

        # act
        sr = compute_satisfaction_rate(distances, d_min)

        # assert - none have rho > 0 (strictly), so SR = 0.0
        assert sr == pytest.approx(0.0)

    def test_mixed_satisfaction(self):
        """Test satisfaction rate with mix of safe and unsafe distances."""
        # arrange
        distances = [15.0, 8.0, 20.0, 5.0, 25.0]
        d_min = 10.0

        # act
        sr = compute_satisfaction_rate(distances, d_min)

        # assert - 3 out of 5 have rho > 0 (15, 20, 25), so SR = 0.6
        assert sr == pytest.approx(0.6)

    def test_boundary_case_exactly_at_dmin(self):
        """Test that distance exactly at d_min (rho=0) does NOT count as satisfied."""
        # arrange - mix including exact boundary
        distances = [10.0, 15.0, 10.0, 20.0]
        d_min = 10.0

        # act
        sr = compute_satisfaction_rate(distances, d_min)

        # assert - only 15.0 and 20.0 have rho > 0, so SR = 2/4 = 0.5
        assert sr == pytest.approx(0.5)

    def test_single_distance_satisfied(self):
        """Test satisfaction rate with single satisfied distance."""
        # arrange
        distances = [15.0]
        d_min = 10.0

        # act
        sr = compute_satisfaction_rate(distances, d_min)

        # assert
        assert sr == pytest.approx(1.0)

    def test_single_distance_not_satisfied(self):
        """Test satisfaction rate with single unsatisfied distance."""
        # arrange
        distances = [8.0]
        d_min = 10.0

        # act
        sr = compute_satisfaction_rate(distances, d_min)

        # assert
        assert sr == pytest.approx(0.0)
