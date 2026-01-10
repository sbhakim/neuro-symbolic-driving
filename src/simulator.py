# src/simulator.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time
import numpy as np

from .vehicle import VehicleState, LeadVehicleProfile, step_dynamics, compute_metrics
from .controllers import AutomationController, HumanController
from .authority_allocator import ClassicalAllocator, LLMOnlyAllocator, NeSyAllocator
from .llm_interface import build_llm


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
class TimingStats:
    """Computational timing statistics (ms)."""
    llm_inference_ms: List[float] = field(default_factory=list)
    monitor_projection_ms: List[float] = field(default_factory=list)
    control_computation_ms: List[float] = field(default_factory=list)
    total_step_ms: List[float] = field(default_factory=list)


def report_timing(timing: TimingStats, dt_ms: float) -> Dict[str, Dict[str, float]]:
    """Summarize timing stats for paper/reporting."""
    def _stats(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"mean_ms": 0.0, "max_ms": 0.0, "std_ms": 0.0}
        return {
            "mean_ms": float(np.mean(xs)),
            "max_ms": float(np.max(xs)),
            "std_ms": float(np.std(xs)),
        }

    total = _stats(timing.total_step_ms)
    total["feasible"] = bool(total["max_ms"] < dt_ms)
    return {
        "llm_inference": _stats(timing.llm_inference_ms),
        "monitor_projection": _stats(timing.monitor_projection_ms),
        "control_computation": _stats(timing.control_computation_ms),
        "total_step": total,
    }


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
    ego_a_min: float = -6.0   # ego min acceleration (max braking) [m/s^2]
    ego_a_max: float = 3.0    # ego max acceleration [m/s^2]

    # LLM backend selection
    llm_backend: str = "mock"        # 'mock' | 'adversarial' | 'openai' | 'hf'
    llm_model: str = "gpt-5-nano"    # OpenAI model name OR HF repo id (when llm_backend='hf')
    llm_period_s: float = 1.0        # call LLM at most once per this many seconds (reduces calls)
    llm_verbose: bool = False        # verbose backend logging (proves OpenAI/HF is used)
    llm_attack_mode: str = "oscillate"   # adversarial attack mode: 'oscillate' | 'force_decrease' | 'random_extreme' | 'confidence_manipulation'
    collect_timing: bool = True      # collect computational timing statistics

    # Authority floor parameters (explicit for paper-friendly experiments)
    llm_only_alpha_min: float = 0.0         # LLM-only floor (set to alpha0 for nominal, 0.0 for adversarial stress test)
    nesy_alpha_floor_emergency: float = 0.85    # NeSy floor during emergency
    nesy_alpha_floor_violation: float = 0.70    # NeSy floor during safety violation
    # nesy_alpha_floor_nominal is always alpha0

    # Scenario controls
    lead_scenario: str = "standard"  # 'standard' | 'severe' | 'repeated'
    stress_test: bool = False        # If True: applies stress-test modifications
    stress_x0_lead_min: float = 50.0         # Minimum initial headway for stress-test
    stress_tau_emerg_min: float = 2.5        # Minimum emergency threshold for stress-test
    stress_force_lead_scenario: str = "repeated"  # Override scenario for stress-test


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
    timing: Optional[TimingStats] = None
    intervention_summary: Optional[Dict[str, Any]] = None
    effective: Optional[Dict[str, Any]] = None  # Resolved effective settings for traceability


def resolve_effective_settings(params: ScenarioParams) -> Dict[str, Any]:
    """
    Resolve effective scenario settings, applying stress-test overrides.

    Returns a dict with resolved values for traceability and consistency.
    """
    eff = {
        "llm_only_alpha_min": float(params.llm_only_alpha_min),
        "lead_scenario": str(params.lead_scenario),
        "x0_lead": float(params.x0_lead),
        "tau_emerg": float(params.tau_emerg),
        "stress_test": bool(params.stress_test),
    }

    if params.stress_test:
        # Override LLM-only floor to 0.0 for true stress test
        eff["llm_only_alpha_min"] = 0.0

        # Force scenario to repeated (multiple emergency onsets) if standard
        if eff["lead_scenario"] == "standard":
            eff["lead_scenario"] = str(params.stress_force_lead_scenario)

        # Increase headway for feasibility
        eff["x0_lead"] = max(eff["x0_lead"], float(params.stress_x0_lead_min))

        # Increase emergency threshold for more interventions
        eff["tau_emerg"] = max(eff["tau_emerg"], float(params.stress_tau_emerg_min))

    return eff


def _make_llm(params: ScenarioParams, rng_mock_llm: np.random.Generator):
    """
    Construct the selected LLM backend via shared factory.

    Returns an object implementing generate_intent(metrics, alpha_current).
    """
    backend = (params.llm_backend or "mock").lower().strip()

    if backend == "adversarial":
        return build_llm(
            mode=backend,
            model=params.llm_model,
            rng=rng_mock_llm,
            verbose=bool(params.llm_verbose),
            attack_mode=getattr(params, "llm_attack_mode", "oscillate"),
        )

    # For other backends, do NOT pass attack_mode (would cause TypeError)
    return build_llm(
        mode=backend,
        model=params.llm_model,
        rng=rng_mock_llm,
        verbose=bool(params.llm_verbose),
    )


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

    # resolve effective settings (applies stress-test overrides)
    eff = resolve_effective_settings(params)

    # print effective settings if verbose
    if params.llm_verbose:
        print(f"[sim:{baseline_type}] effective settings: {eff}")

    # initialize vehicles
    ego = VehicleState(x=params.x0_ego, v=params.v0_ego, a=0.0, t=0.0)
    lead = VehicleState(x=eff["x0_lead"], v=params.v0_lead, a=0.0, t=0.0)
    lead_profile = LeadVehicleProfile(scenario=eff["lead_scenario"], ego_a_min=params.ego_a_min)

    # store lead profile description in effective settings
    eff["lead_profile"] = lead_profile.describe()

    # create controllers
    auto_controller = AutomationController()
    human_controller = HumanController(dt=params.dt, rng=rng_human)

    # create authority allocator based on baseline
    if baseline_type == 'classical':
        allocator = ClassicalAllocator(rng=rng_classical)
        llm = None
    elif baseline_type == 'llm_only':
        # Use resolved alpha_min from effective settings
        allocator = LLMOnlyAllocator(eta=params.eta, alpha_min=float(eff["llm_only_alpha_min"]))
        llm = _make_llm(params, rng_mock_llm)
    elif baseline_type == 'nesy':
        # Pass explicit floor parameters for paper-friendly experiments
        allocator = NeSyAllocator(
            eta=params.eta,
            gamma=params.gamma,
            dt=params.dt,
            alpha_floor_emergency=float(params.nesy_alpha_floor_emergency),
            alpha_floor_violation=float(params.nesy_alpha_floor_violation),
            alpha_floor_nominal=float(params.alpha0),
        )
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

    # initialize timing collection
    timing = TimingStats() if getattr(params, "collect_timing", False) else None

    # simulation loop
    for k in range(num_steps):
        step_t0 = time.perf_counter() if timing is not None else None

        # compute safety metrics and controls
        ctrl_t0 = time.perf_counter() if timing is not None else None

        metrics = compute_metrics(ego, lead, d_min=params.d_min, tau_emerg=eff["tau_emerg"])

        # compute automation control
        u_auto = auto_controller.compute_control(ego, lead)

        # compute human control
        u_human = human_controller.compute_control(u_auto, metrics.emergency)

        # blend controls
        u_blend = alpha * u_auto + (1.0 - alpha) * u_human

        if timing is not None and ctrl_t0 is not None:
            timing.control_computation_ms.append((time.perf_counter() - ctrl_t0) * 1000.0)

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
                if timing is not None:
                    llm_t0 = time.perf_counter()
                    last_intent = llm.generate_intent(metrics, alpha)
                    timing.llm_inference_ms.append((time.perf_counter() - llm_t0) * 1000.0)
                else:
                    last_intent = llm.generate_intent(metrics, alpha)

            intent = last_intent
            if baseline_type == 'llm_only':
                alpha = allocator.update(alpha, intent)
            else:  # nesy
                if timing is not None:
                    proj_t0 = time.perf_counter()
                    alpha = allocator.update(alpha, intent, metrics.emergency, metrics.robustness, step=k)
                    timing.monitor_projection_ms.append((time.perf_counter() - proj_t0) * 1000.0)
                else:
                    alpha = allocator.update(alpha, intent, metrics.emergency, metrics.robustness, step=k)

            prev_emergency = metrics.emergency

        # update ego dynamics
        w = rng_ego.normal(0.0, params.sigma_w)
        ego = step_dynamics(
            ego, u_blend, params.dt, w=w,
            a_min=float(params.ego_a_min),
            a_max=float(params.ego_a_max)
        )

        # update lead dynamics with exact scripted acceleration
        # Use lead.t (current step time), not ego.t (already advanced)
        a_lead = lead_profile.get_acceleration(lead.t, lead.v)
        lead = step_with_accel(lead, a_lead, params.dt)

        # record total step time
        if timing is not None and step_t0 is not None:
            timing.total_step_ms.append((time.perf_counter() - step_t0) * 1000.0)

    # attach timing and report if verbose
    traj.timing = timing

    if timing is not None and params.llm_verbose:
        summary = report_timing(timing, dt_ms=params.dt * 1000.0)
        print(f"[Timing {baseline_type}]", summary)

    # attach intervention summary for NeSy baseline
    if baseline_type == 'nesy':
        traj.intervention_summary = allocator.get_intervention_summary()

    # attach effective settings for traceability
    traj.effective = eff

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
