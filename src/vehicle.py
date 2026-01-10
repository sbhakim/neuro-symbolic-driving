# src/vehicle.py

from dataclasses import dataclass

import numpy as np


@dataclass
class VehicleState:
    """Complete vehicle state at discrete time k."""
    x: float  # position [m]
    v: float  # velocity [m/s]
    a: float  # acceleration [m/s^2]
    t: float  # time [s]


@dataclass
class SafetyMetrics:
    """Safety metrics computed from ego and lead vehicle states."""
    distance: float       # headway d = x_lead - x_ego [m]
    relative_v: float     # v_ego - v_lead [m/s]
    ttc: float            # time-to-collision [s]
    emergency: bool       # TTC <= tau_emerg
    robustness: float     # rho = d - d_min [m]


def step_dynamics(
    state: VehicleState,
    u: float,
    dt: float,
    w: float = 0.0,
    a_min: float = -6.0,
    a_max: float = 3.0
) -> VehicleState:
    """
    Update vehicle state using discrete-time dynamics.

    x_{k+1} = x_k + v_k * dt
    v_{k+1} = max(0, v_k + a_k * dt)
    a_k = clip(u_k + w_k, a_min, a_max)
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")

    a = float(np.clip(u + w, a_min, a_max))
    v_next = max(0.0, state.v + a * dt)
    x_next = state.x + state.v * dt
    t_next = state.t + dt
    return VehicleState(x=x_next, v=v_next, a=a, t=t_next)


class LeadVehicleProfile:
    """Scripted lead vehicle braking profile with recovery for emergency scenario."""

    def __init__(
        self,
        scenario: str = "standard",
        v_cruise: float = 30.0,
        recover_accel: float = 2.5,
        ego_a_min: float = -6.0,
        feasibility_tolerance: float = 0.6
    ):
        """
        Args:
            scenario: 'standard' | 'severe' | 'repeated'
            v_cruise: Target cruise velocity [m/s]
            recover_accel: Recovery acceleration [m/s^2]
            ego_a_min: Ego min acceleration (for feasibility check) [m/s^2]
            feasibility_tolerance: Allowed margin for lead braking vs ego capability [m/s^2]
        """
        self.scenario = scenario.lower().strip()
        self.v_cruise = v_cruise
        self.recover_accel = recover_accel

        # Configure scenario-specific parameters
        if self.scenario == "standard":
            # Single braking event: moderate intensity, moderate duration
            self.brake_accel = -6.5
            self.brake_intervals = [(10.0, 14.0)]
        elif self.scenario == "severe":
            # Single braking event: longer duration, ego-feasible intensity
            # Severity comes from extended braking time, not exceeding ego limits
            self.brake_accel = -6.0  # equal to ego a_min magnitude (feasible)
            self.brake_intervals = [(10.0, 15.0)]
        elif self.scenario == "repeated":
            # Multiple braking pulses: repeated emergency onsets for stress-testing
            self.brake_accel = -6.5
            self.brake_intervals = [(8.0, 11.0), (16.0, 19.0), (24.0, 27.0)]
        else:
            raise ValueError(f"Unknown scenario: {scenario} (expected: standard|severe|repeated)")

        # Guardrail: prevent significantly infeasible scenarios
        # Tolerance allows marginal cases that are feasible with proper headway
        if self.brake_accel < (ego_a_min - feasibility_tolerance):
            raise ValueError(
                f"Infeasible: lead brake_accel={self.brake_accel} significantly exceeds "
                f"ego_a_min={ego_a_min} (tolerance={feasibility_tolerance}). "
                "Either increase ego braking authority or reduce lead braking."
            )

    def get_acceleration(self, t: float, v_lead: float) -> float:
        """
        Return scripted acceleration at time t based on lead velocity.

        Phases:
        - If t in any brake_interval: brake (a=brake_accel)
        - Else if v_lead < v_cruise: recover (a=recover_accel)
        - Else: cruise (a=0)
        """
        # Check if currently in any braking interval
        for brake_start, brake_end in self.brake_intervals:
            if brake_start <= t < brake_end:
                return float(self.brake_accel)

        # Not braking: recover if below cruise speed, else cruise
        if v_lead < self.v_cruise:
            return float(self.recover_accel)
        return 0.0

    def describe(self) -> dict:
        """
        Return configuration dict for logging and traceability.

        Returns dict with scenario parameters for inclusion in metrics/outputs.
        """
        return {
            "scenario": self.scenario,
            "v_cruise": float(self.v_cruise),
            "recover_accel": float(self.recover_accel),
            "brake_accel": float(self.brake_accel),
            "brake_intervals": list(self.brake_intervals),
        }


def compute_metrics(
    ego: VehicleState,
    lead: VehicleState,
    d_min: float = 10.0,
    tau_emerg: float = 2.0,
    epsilon: float = 0.1
) -> SafetyMetrics:
    """
    Compute safety metrics from ego and lead vehicle states.

    d = x_lead - x_ego
    rel_v = v_ego - v_lead
    TTC = d / max(rel_v, eps) if rel_v > 0 else +inf
    Robustness rho = d - d_min
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    distance = float(lead.x - ego.x)
    relative_v = float(ego.v - lead.v)

    # TTC should not go negative; if distance <= 0 we treat it as collision imminent (TTC = 0).
    if distance <= 0.0:
        ttc = 0.0
    elif relative_v > 0.0:
        ttc = distance / max(relative_v, epsilon)
    else:
        ttc = float("inf")

    emergency = bool(ttc <= tau_emerg)
    robustness = float(distance - d_min)

    return SafetyMetrics(
        distance=distance,
        relative_v=relative_v,
        ttc=float(ttc),
        emergency=emergency,
        robustness=robustness,
    )
