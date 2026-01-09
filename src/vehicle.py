# src/vehicle.py

from dataclasses import dataclass
from typing import Final

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

    def __init__(self, brake_start: float = 10.0, brake_end: float = 14.0,
                 brake_accel: float = -6.5, v_cruise: float = 30.0,
                 recover_accel: float = 2.5):
        """
        Args:
            brake_start: Time when braking begins [s]
            brake_end: Time when braking ends [s]
            brake_accel: Braking acceleration [m/s^2]
            v_cruise: Target cruise velocity [m/s]
            recover_accel: Recovery acceleration [m/s^2]
        """
        self.brake_start = brake_start
        self.brake_end = brake_end
        self.brake_accel = brake_accel
        self.v_cruise = v_cruise
        self.recover_accel = recover_accel

    def get_acceleration(self, t: float, v_lead: float) -> float:
        """
        Return scripted acceleration at time t based on lead velocity.

        Phases:
        - t < brake_start: cruise (a=0)
        - brake_start <= t < brake_end: brake (a=brake_accel)
        - t >= brake_end: recover until v_lead >= v_cruise, then cruise
        """
        if t < self.brake_start:
            return 0.0
        elif t < self.brake_end:
            return float(self.brake_accel)
        else:
            # recovery phase: accelerate until back at cruise speed
            if v_lead < self.v_cruise:
                return float(self.recover_accel)
            else:
                return 0.0


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
    robustness = distance - d_min

    return SafetyMetrics(
        distance=distance,
        relative_v=relative_v,
        ttc=float(ttc),
        emergency=emergency,
        robustness=float(robustness),
    )
