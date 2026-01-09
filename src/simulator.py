# src/simulator.py

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from .vehicle import VehicleState, LeadVehicleProfile, step_dynamics, compute_metrics
from .controllers import AutomationController, HumanController
from .authority_allocator import ClassicalAllocator, LLMOnlyAllocator, NeSyAllocator
from .llm_interface import MockLLM, RealLLM, OfflineHFLLM


def step_with_accel(state: VehicleState, a: float, dt: float) -> VehicleState:
    """
    Update vehicle state with exact acceleration (no clipping for scripted lead).

    x_{k+1} = x_k + v_k * dt
    v_{k+1} = max(0, v_k + a * dt)
    """
    v_next = max(0.0, state.v + a * dt)
    x_next = state.x + state.v * dt
    t_next = state.t + dt
    return VehicleState(x=x_next, v=v_next, a=a, t=t_next)


@dataclass
class ScenarioParams:
    """Simulation scenario parameters."""
    dt: float = 0.1           # sampling period [s]
    T: float = 30.0           # horizon [s]
    x0_ego: float = 0.0       # initial ego position [m]
    v0_ego: float = 30.0      # initial ego velocity [m/s]
    x0_lead: float = 30.0     # initial lead position [m]
    v0_lead: float = 30.0     # initial lead velocity [m/s]
    alpha0: float = 0.2       # initial authority
    d_min: float = 10.0       # minimum safe distance [m]
    tau_emerg: float = 2.0    # emergency TTC threshold [s]
    gamma: float = 0.5        # max authority rate [1/s]
    eta: float = 0.3          # proposal inertia
    sigma_w: float = 0.10     # ego process noise std [m/s^2]

    # LLM backend selection
    llm_backend: str = "mock"        # 'mock' | 'openai' | 'hf'
    llm_model: str = "gpt-5-nano"    # OpenAI model name OR HF repo id (when llm_backend='hf')
    llm_period_s: float = 1.0        # call LLM at most once per this many seconds (reduces calls)
    llm_verbose: bool = False        # verbose backend logging (proves OpenAI/HF is used)


@dataclass
class Trajectory:
    """Time-series trajectory data."""
    t: List[float]
    x_ego: List[float]
    v_ego: List[float]
    a_ego: List[float]
    x_lead: List[float]
    v_lead: List[float]
    a_lead: List[float]
    distance: List[float]
    ttc: List[float]
    emergency: List[bool]
    robustness: List[float]
    alpha: List[float]
    u_auto: List[float]
    u_human: List[float]
    u_blend: List[float]


def _make_llm(params: ScenarioParams, rng_mock_llm: np.random.Generator):
    """
    Construct the selected LLM backend (mock/openai/hf).

    Returns an object implementing generate_intent(metrics, alpha_current).
    """
    backend = (params.llm_backend or "mock").lower().strip()

    if backend == "openai":
        return RealLLM(model=params.llm_model, verbose=bool(params.llm_verbose))

    if backend == "hf":
        return OfflineHFLLM(model_id=params.llm_model, verbose=bool(params.llm_verbose))

    # default: mock
    return MockLLM(rng=rng_mock_llm)


def run_baseline(baseline_type: str, seed: int,
                 params: ScenarioParams) -> Trajectory:
    """
    Run a single baseline scenario.

    Args:
        baseline_type: 'classical', 'llm_only', or 'nesy'
        seed: Random seed for reproducibility
        params: Scenario parameters

    Returns:
        Trajectory with time-series data
    """
    # create RNGs from seed
    rng_ego = np.random.default_rng(seed)
    rng_classical = np.random.default_rng(seed + 1)
    rng_mock_llm = np.random.default_rng(seed + 2)
    rng_human = np.random.default_rng(seed + 3)

    # initialize vehicles
    ego = VehicleState(x=params.x0_ego, v=params.v0_ego, a=0.0, t=0.0)
    lead = VehicleState(x=params.x0_lead, v=params.v0_lead, a=0.0, t=0.0)
    lead_profile = LeadVehicleProfile()

    # create controllers
    auto_controller = AutomationController()
    human_controller = HumanController(dt=params.dt, rng=rng_human)

    # create authority allocator based on baseline
    if baseline_type == 'classical':
        allocator = ClassicalAllocator(rng=rng_classical)
        llm = None
    elif baseline_type == 'llm_only':
        allocator = LLMOnlyAllocator(eta=params.eta)
        llm = _make_llm(params, rng_mock_llm)
    elif baseline_type == 'nesy':
        allocator = NeSyAllocator(eta=params.eta, gamma=params.gamma, dt=params.dt)
        llm = _make_llm(params, rng_mock_llm)
    else:
        raise ValueError(f"Unknown baseline: {baseline_type}")

    # simulation state
    alpha = params.alpha0
    num_steps = int(params.T / params.dt)

    # LLM call throttling (prevents "stuck" behavior from too many calls)
    llm_every = max(1, int(round(max(params.llm_period_s, params.dt) / params.dt)))
    last_intent = None
    prev_emergency = False

    # initialize trajectory storage
    traj = Trajectory(
        t=[], x_ego=[], v_ego=[], a_ego=[],
        x_lead=[], v_lead=[], a_lead=[],
        distance=[], ttc=[], emergency=[], robustness=[],
        alpha=[], u_auto=[], u_human=[], u_blend=[]
    )

    # simulation loop
    for k in range(num_steps):
        # compute safety metrics
        metrics = compute_metrics(ego, lead, d_min=params.d_min, tau_emerg=params.tau_emerg)

        # compute automation control
        u_auto = auto_controller.compute_control(ego, lead)

        # compute human control
        u_human = human_controller.compute_control(u_auto, metrics.emergency)

        # blend controls
        u_blend = alpha * u_auto + (1.0 - alpha) * u_human

        # log current state
        traj.t.append(ego.t)
        traj.x_ego.append(ego.x)
        traj.v_ego.append(ego.v)
        traj.a_ego.append(ego.a)
        traj.x_lead.append(lead.x)
        traj.v_lead.append(lead.v)
        traj.a_lead.append(lead.a)
        traj.distance.append(metrics.distance)
        traj.ttc.append(metrics.ttc)
        traj.emergency.append(metrics.emergency)
        traj.robustness.append(metrics.robustness)
        traj.alpha.append(alpha)
        traj.u_auto.append(u_auto)
        traj.u_human.append(u_human)
        traj.u_blend.append(u_blend)

        # update authority for next step
        if baseline_type == 'classical':
            alpha = allocator.update(alpha, metrics.ttc, metrics.distance)
        else:  # llm_only or nesy
            emergency_onset = (metrics.emergency and not prev_emergency)
            if last_intent is None or (k % llm_every == 0) or emergency_onset:
                last_intent = llm.generate_intent(metrics, alpha)

            intent = last_intent
            if baseline_type == 'llm_only':
                alpha = allocator.update(alpha, intent)
            else:  # nesy
                alpha = allocator.update(alpha, intent, metrics.emergency, metrics.robustness)

            prev_emergency = metrics.emergency

        # update ego dynamics
        w = rng_ego.normal(0.0, params.sigma_w)
        ego = step_dynamics(ego, u_blend, params.dt, w=w)

        # update lead dynamics with exact scripted acceleration
        a_lead = lead_profile.get_acceleration(ego.t, lead.v)
        lead = step_with_accel(lead, a_lead, params.dt)

    return traj


def run_all_baselines(seed: int = 0,
                      params: ScenarioParams = None) -> Dict[str, Trajectory]:
    """
    Run all three baselines with the same scenario and seed.

    Args:
        seed: Random seed for reproducibility
        params: Scenario parameters (uses defaults if None)

    Returns:
        Dictionary mapping baseline names to trajectories
    """
    if params is None:
        params = ScenarioParams()

    results = {}
    for baseline in ['classical', 'llm_only', 'nesy']:
        results[baseline] = run_baseline(baseline, seed, params)

    return results
