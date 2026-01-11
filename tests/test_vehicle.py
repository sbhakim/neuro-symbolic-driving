# tests/test_vehicle.py

import pytest

from src.vehicle import VehicleState, step_dynamics, compute_metrics, LeadVehicleProfile


class TestStepDynamics:
    """Unit tests for vehicle dynamics update."""

    def test_basic_euler_update(self):
        """Test basic Euler integration: x_next = x + v*dt, v_next = v + a*dt."""
        # arrange
        state = VehicleState(x=0.0, v=10.0, a=0.0, t=0.0)
        u = 2.0
        dt = 0.1

        # act
        next_state = step_dynamics(state, u, dt, w=0.0)

        # assert
        assert next_state.x == pytest.approx(0.0 + 10.0 * 0.1)  # x + v*dt = 1.0
        assert next_state.v == pytest.approx(10.0 + 2.0 * 0.1)  # v + a*dt = 10.2
        assert next_state.a == pytest.approx(2.0)
        assert next_state.t == pytest.approx(0.1)

    def test_velocity_non_negative(self):
        """Test that velocity is clipped to non-negative."""
        # arrange
        state = VehicleState(x=10.0, v=0.1, a=0.0, t=0.0)
        u = -20.0  # large deceleration (clipped to -6.0)
        dt = 0.1

        # act
        next_state = step_dynamics(state, u, dt, w=0.0)

        # assert - velocity should be clipped to 0 (0.1 + (-6.0)*0.1 = -0.5 -> 0.0)
        assert next_state.v == pytest.approx(0.0)
        assert next_state.v >= 0.0

    def test_acceleration_clipping_upper_bound(self):
        """Test acceleration is clipped to upper bound a_max=3.0."""
        # arrange
        state = VehicleState(x=0.0, v=10.0, a=0.0, t=0.0)
        u = 10.0  # exceeds a_max
        dt = 0.1

        # act
        next_state = step_dynamics(state, u, dt, w=0.0, a_min=-6.0, a_max=3.0)

        # assert - acceleration clipped to 3.0
        assert next_state.a == pytest.approx(3.0)

    def test_acceleration_clipping_lower_bound(self):
        """Test acceleration is clipped to lower bound a_min=-6.0."""
        # arrange
        state = VehicleState(x=0.0, v=10.0, a=0.0, t=0.0)
        u = -10.0  # exceeds a_min
        dt = 0.1

        # act
        next_state = step_dynamics(state, u, dt, w=0.0, a_min=-6.0, a_max=3.0)

        # assert - acceleration clipped to -6.0
        assert next_state.a == pytest.approx(-6.0)

    def test_process_noise_added(self):
        """Test that process noise w is added to control input."""
        # arrange
        state = VehicleState(x=0.0, v=10.0, a=0.0, t=0.0)
        u = 1.0
        w = 0.5
        dt = 0.1

        # act
        next_state = step_dynamics(state, u, dt, w=w, a_min=-6.0, a_max=3.0)

        # assert - effective acceleration is u + w = 1.5
        assert next_state.a == pytest.approx(1.5)

    def test_invalid_dt_raises_error(self):
        """Test that dt <= 0 raises ValueError."""
        # arrange
        state = VehicleState(x=0.0, v=10.0, a=0.0, t=0.0)

        # act & assert - zero dt
        with pytest.raises(ValueError, match="dt must be > 0"):
            step_dynamics(state, u=0.0, dt=0.0)

        # act & assert - negative dt
        with pytest.raises(ValueError, match="dt must be > 0"):
            step_dynamics(state, u=0.0, dt=-0.1)


class TestComputeMetrics:
    """Unit tests for safety metrics computation."""

    def test_relative_velocity_zero_ttc_infinite(self):
        """Test when relative_v <= 0 and distance > 0, TTC is infinite and no emergency."""
        # arrange - vehicles at same speed, ego behind lead
        ego = VehicleState(x=0.0, v=20.0, a=0.0, t=0.0)
        lead = VehicleState(x=30.0, v=20.0, a=0.0, t=0.0)

        # act
        metrics = compute_metrics(ego, lead, d_min=10.0, tau_emerg=2.0)

        # assert
        assert metrics.distance == pytest.approx(30.0)
        assert metrics.relative_v == pytest.approx(0.0)
        assert metrics.ttc == float("inf")
        assert metrics.emergency is False

    def test_relative_velocity_negative_ttc_infinite(self):
        """Test when ego slower than lead (relative_v < 0), TTC is infinite."""
        # arrange - ego slower, diverging
        ego = VehicleState(x=0.0, v=15.0, a=0.0, t=0.0)
        lead = VehicleState(x=30.0, v=25.0, a=0.0, t=0.0)

        # act
        metrics = compute_metrics(ego, lead, d_min=10.0, tau_emerg=2.0)

        # assert
        assert metrics.relative_v == pytest.approx(-10.0)  # ego slower
        assert metrics.ttc == float("inf")
        assert metrics.emergency is False

    def test_distance_zero_ttc_zero_emergency(self):
        """Test when distance <= 0 (collision), TTC is 0 and emergency is True."""
        # arrange - collision scenario
        ego = VehicleState(x=30.0, v=25.0, a=0.0, t=0.0)
        lead = VehicleState(x=30.0, v=20.0, a=0.0, t=0.0)

        # act
        metrics = compute_metrics(ego, lead, d_min=10.0, tau_emerg=2.0)

        # assert
        assert metrics.distance == pytest.approx(0.0)
        assert metrics.ttc == pytest.approx(0.0)
        assert metrics.emergency is True  # TTC=0 <= tau_emerg

    def test_distance_negative_ttc_zero(self):
        """Test when distance < 0 (past collision), TTC is 0."""
        # arrange - ego overtook lead
        ego = VehicleState(x=35.0, v=25.0, a=0.0, t=0.0)
        lead = VehicleState(x=30.0, v=20.0, a=0.0, t=0.0)

        # act
        metrics = compute_metrics(ego, lead, d_min=10.0, tau_emerg=2.0)

        # assert
        assert metrics.distance == pytest.approx(-5.0)
        assert metrics.ttc == pytest.approx(0.0)

    def test_ttc_calculation_approaching(self):
        """Test TTC = distance / max(relative_v, epsilon) when approaching."""
        # arrange - ego approaching lead
        ego = VehicleState(x=0.0, v=30.0, a=0.0, t=0.0)
        lead = VehicleState(x=20.0, v=20.0, a=0.0, t=0.0)
        epsilon = 0.1

        # act
        metrics = compute_metrics(ego, lead, d_min=10.0, tau_emerg=2.0, epsilon=epsilon)

        # assert
        expected_ttc = 20.0 / max(10.0, epsilon)  # d / delta_v = 20 / 10 = 2.0
        assert metrics.distance == pytest.approx(20.0)
        assert metrics.relative_v == pytest.approx(10.0)
        assert metrics.ttc == pytest.approx(expected_ttc)

    def test_ttc_below_threshold_triggers_emergency(self):
        """Test emergency flag is True when TTC <= tau_emerg."""
        # arrange - approaching with low TTC
        ego = VehicleState(x=0.0, v=25.0, a=0.0, t=0.0)
        lead = VehicleState(x=15.0, v=20.0, a=0.0, t=0.0)
        tau_emerg = 3.0

        # act
        metrics = compute_metrics(ego, lead, d_min=10.0, tau_emerg=tau_emerg)

        # assert - TTC = 15 / 5 = 3.0, exactly at threshold
        assert metrics.ttc == pytest.approx(3.0)
        assert metrics.emergency is True

    def test_ttc_above_threshold_no_emergency(self):
        """Test emergency flag is False when TTC > tau_emerg."""
        # arrange - safe following
        ego = VehicleState(x=0.0, v=25.0, a=0.0, t=0.0)
        lead = VehicleState(x=50.0, v=20.0, a=0.0, t=0.0)
        tau_emerg = 2.0

        # act
        metrics = compute_metrics(ego, lead, d_min=10.0, tau_emerg=tau_emerg)

        # assert - TTC = 50 / 5 = 10.0 >> tau_emerg
        assert metrics.ttc == pytest.approx(10.0)
        assert metrics.emergency is False

    def test_robustness_calculation(self):
        """Test robustness = distance - d_min."""
        # arrange
        ego = VehicleState(x=0.0, v=20.0, a=0.0, t=0.0)
        lead = VehicleState(x=25.0, v=20.0, a=0.0, t=0.0)
        d_min = 10.0

        # act
        metrics = compute_metrics(ego, lead, d_min=d_min, tau_emerg=2.0)

        # assert
        assert metrics.robustness == pytest.approx(25.0 - 10.0)  # 15.0

    def test_invalid_epsilon_raises_error(self):
        """Test that epsilon <= 0 raises ValueError."""
        # arrange
        ego = VehicleState(x=0.0, v=25.0, a=0.0, t=0.0)
        lead = VehicleState(x=30.0, v=20.0, a=0.0, t=0.0)

        # act & assert - zero epsilon
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            compute_metrics(ego, lead, epsilon=0.0)

        # act & assert - negative epsilon
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            compute_metrics(ego, lead, epsilon=-0.1)


class TestLeadVehicleProfile:
    """Unit tests for scenario-based lead vehicle profile."""

    def test_standard_scenario_configuration(self):
        """Test standard scenario has correct brake parameters."""
        # arrange & act
        profile = LeadVehicleProfile(scenario="standard")

        # assert
        assert profile.scenario == "standard"
        assert profile.brake_accel == pytest.approx(-6.5)
        assert profile.brake_intervals == [(10.0, 14.0)]
        assert profile.v_cruise == pytest.approx(30.0)
        assert profile.recover_accel == pytest.approx(2.5)

    def test_severe_scenario_configuration(self):
        """Test severe scenario has longer braking duration."""
        # arrange & act
        profile = LeadVehicleProfile(scenario="severe")

        # assert
        assert profile.scenario == "severe"
        assert profile.brake_accel == pytest.approx(-6.0)
        assert profile.brake_intervals == [(10.0, 15.0)]

    def test_repeated_scenario_configuration(self):
        """Test repeated scenario has multiple braking pulses."""
        # arrange & act
        profile = LeadVehicleProfile(scenario="repeated")

        # assert
        assert profile.scenario == "repeated"
        assert profile.brake_accel == pytest.approx(-6.5)
        assert profile.brake_intervals == [(8.0, 11.0), (16.0, 19.0), (24.0, 27.0)]

    def test_invalid_scenario_raises_error(self):
        """Test that invalid scenario name raises ValueError."""
        # act & assert
        with pytest.raises(ValueError, match="Unknown scenario"):
            LeadVehicleProfile(scenario="invalid")

    def test_cruise_phase_before_brake(self):
        """Test cruise phase returns zero acceleration when at cruise speed."""
        # arrange
        profile = LeadVehicleProfile(scenario="standard")

        # act & assert - before braking interval
        assert profile.get_acceleration(t=5.0, v_lead=30.0) == pytest.approx(0.0)

    def test_brake_phase_standard(self):
        """Test brake phase returns brake_accel during brake interval."""
        # arrange
        profile = LeadVehicleProfile(scenario="standard")

        # act & assert - during braking (10.0 <= t < 14.0)
        assert profile.get_acceleration(t=10.0, v_lead=30.0) == pytest.approx(-6.5)
        assert profile.get_acceleration(t=12.0, v_lead=25.0) == pytest.approx(-6.5)
        assert profile.get_acceleration(t=13.9, v_lead=20.0) == pytest.approx(-6.5)

    def test_brake_phase_boundary_end(self):
        """Test brake phase ends exactly at brake_end (exclusive)."""
        # arrange
        profile = LeadVehicleProfile(scenario="standard")

        # act & assert - t=14.0 is NOT in [10.0, 14.0), should recover
        a = profile.get_acceleration(t=14.0, v_lead=25.0)
        assert a == pytest.approx(2.5)  # recovery, not braking

    def test_brake_phase_repeated_multiple_intervals(self):
        """Test repeated scenario brakes during all three intervals."""
        # arrange
        profile = LeadVehicleProfile(scenario="repeated")

        # act & assert - first interval (8.0 <= t < 11.0)
        assert profile.get_acceleration(t=9.0, v_lead=30.0) == pytest.approx(-6.5)

        # second interval (16.0 <= t < 19.0)
        assert profile.get_acceleration(t=17.0, v_lead=25.0) == pytest.approx(-6.5)

        # third interval (24.0 <= t < 27.0)
        assert profile.get_acceleration(t=25.0, v_lead=20.0) == pytest.approx(-6.5)

    def test_recovery_phase_below_cruise(self):
        """Test recovery phase returns recover_accel when v < v_cruise."""
        # arrange
        profile = LeadVehicleProfile(scenario="standard", v_cruise=30.0, recover_accel=2.5)

        # act & assert - after braking, velocity below cruise
        assert profile.get_acceleration(t=15.0, v_lead=20.0) == pytest.approx(2.5)
        assert profile.get_acceleration(t=20.0, v_lead=28.0) == pytest.approx(2.5)

    def test_recovery_phase_at_cruise(self):
        """Test recovery phase returns zero when v >= v_cruise."""
        # arrange
        profile = LeadVehicleProfile(scenario="standard", v_cruise=30.0)

        # act & assert - velocity at or above cruise speed
        assert profile.get_acceleration(t=20.0, v_lead=30.0) == pytest.approx(0.0)
        assert profile.get_acceleration(t=20.0, v_lead=31.0) == pytest.approx(0.0)

    def test_recovery_between_repeated_intervals(self):
        """Test recovery occurs between repeated braking intervals."""
        # arrange
        profile = LeadVehicleProfile(scenario="repeated")

        # act & assert - between first and second interval (t=12, v<30)
        assert profile.get_acceleration(t=12.0, v_lead=25.0) == pytest.approx(2.5)

        # between second and third interval (t=20, v<30)
        assert profile.get_acceleration(t=20.0, v_lead=22.0) == pytest.approx(2.5)

    def test_feasibility_guardrail_accepts_valid_scenario(self):
        """Test feasibility check passes for ego-feasible scenarios."""
        # arrange & act - standard scenario: brake=-6.5, ego_a_min=-6.0, tolerance=0.6
        # -6.5 > -6.0 - 0.6 = -6.6, so feasible
        profile = LeadVehicleProfile(scenario="standard", ego_a_min=-6.0)

        # assert - should not raise
        assert profile.brake_accel == pytest.approx(-6.5)

    def test_feasibility_guardrail_rejects_infeasible_scenario(self):
        """Test feasibility check rejects significantly infeasible scenarios."""
        # arrange & act & assert
        # brake=-6.5, ego_a_min=-5.0, tolerance=0.6
        # -6.5 < -5.0 - 0.6 = -5.6, so infeasible
        with pytest.raises(ValueError, match="Infeasible"):
            LeadVehicleProfile(scenario="standard", ego_a_min=-5.0, feasibility_tolerance=0.6)

    def test_feasibility_guardrail_with_custom_tolerance(self):
        """Test feasibility check respects custom tolerance."""
        # arrange & act
        # brake=-6.5, ego_a_min=-6.0, tolerance=1.0
        # -6.5 > -6.0 - 1.0 = -7.0, so feasible with larger tolerance
        profile = LeadVehicleProfile(
            scenario="standard",
            ego_a_min=-6.0,
            feasibility_tolerance=1.0
        )

        # assert - should not raise
        assert profile.brake_accel == pytest.approx(-6.5)

    def test_describe_method_standard(self):
        """Test describe() returns complete configuration dict for standard scenario."""
        # arrange
        profile = LeadVehicleProfile(scenario="standard", v_cruise=30.0, recover_accel=2.5)

        # act
        desc = profile.describe()

        # assert
        assert desc["scenario"] == "standard"
        assert desc["v_cruise"] == pytest.approx(30.0)
        assert desc["recover_accel"] == pytest.approx(2.5)
        assert desc["brake_accel"] == pytest.approx(-6.5)
        assert desc["brake_intervals"] == [(10.0, 14.0)]

    def test_describe_method_repeated(self):
        """Test describe() captures all brake intervals for repeated scenario."""
        # arrange
        profile = LeadVehicleProfile(scenario="repeated")

        # act
        desc = profile.describe()

        # assert
        assert desc["scenario"] == "repeated"
        assert desc["brake_intervals"] == [(8.0, 11.0), (16.0, 19.0), (24.0, 27.0)]
        assert desc["brake_accel"] == pytest.approx(-6.5)
