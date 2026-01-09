# src/controllers.py

from collections import deque
from typing import Deque, Optional
import numpy as np
from .vehicle import VehicleState


class AutomationController:
    """PD car-following controller for automation."""

    def __init__(self, kp: float = 0.30, kd: float = 0.80,
                 d0: float = 5.0, tau_h: float = 0.83,
                 a_min: float = -6.0, a_max: float = 3.0):
        """
        Args:
            kp: Distance error gain
            kd: Relative velocity gain
            d0: Standstill distance [m]
            tau_h: Time headway [s] (cruise ~30m at 30m/s)
            a_min, a_max: Acceleration limits [m/s^2]
        """
        self.kp = kp
        self.kd = kd
        self.d0 = d0
        self.tau_h = tau_h
        self.a_min = a_min
        self.a_max = a_max

    def compute_control(self, ego: VehicleState, lead: VehicleState) -> float:
        """
        PD car-following control.

        d_target = d_0 + tau_h * v_ego
        e_d = d - d_target
        e_v = v_ego - v_lead
        u_A = K_p * e_d - K_d * e_v
        """
        distance = lead.x - ego.x
        d_target = self.d0 + self.tau_h * ego.v
        e_d = distance - d_target
        e_v = ego.v - lead.v
        u = self.kp * e_d - self.kd * e_v

        return np.clip(u, self.a_min, self.a_max)


class HumanController:
    """Human controller with delay, noise, and panic response."""

    def __init__(self, dt: float, tau_delay: float = 0.3,
                 sigma_h: float = 0.3, beta_panic: float = 0.2,
                 a_min: float = -6.0, a_max: float = 3.0,
                 rng: Optional[np.random.Generator] = None):
        """
        Args:
            dt: Sampling period [s]
            tau_delay: Human reaction delay [s]
            sigma_h: Control noise std [m/s^2]
            beta_panic: Emergency overreaction factor
            a_min, a_max: Acceleration limits [m/s^2]
            rng: Random number generator for reproducibility
        """
        self.dt = dt
        self.sigma_h = sigma_h
        self.beta_panic = beta_panic
        self.a_min = a_min
        self.a_max = a_max
        self.rng = rng if rng is not None else np.random.default_rng()

        # delay buffer stores past automation commands
        self.delay_steps = max(1, int(round(tau_delay / dt)))
        self.auto_buffer: Deque[float] = deque(maxlen=self.delay_steps + 1)
        # initialize with zeros
        for _ in range(self.delay_steps + 1):
            self.auto_buffer.append(0.0)

    def compute_control(self, u_auto: float, emergency: bool) -> float:
        """
        Delayed + noisy human control with panic response.

        u_H = (1 + beta_panic) * u_{k-delta}^A + N(0, sigma_h^2)  if emergency
            = u_{k-delta}^A + N(0, sigma_h^2)                      otherwise
        """
        self.auto_buffer.append(u_auto)
        u_delayed = self.auto_buffer[0]

        if emergency:
            u = (1.0 + self.beta_panic) * u_delayed
        else:
            u = u_delayed

        # add noise
        noise = self.rng.normal(0.0, self.sigma_h)
        u += noise

        return np.clip(u, self.a_min, self.a_max)
