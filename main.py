# main.py

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

from src.simulator import run_all_baselines, ScenarioParams, Trajectory
from src.visualization import plot_four_panel_figure, save_figure
from src.stl_monitor import compute_rho_min, compute_satisfaction_rate


def compute_metrics(traj: Trajectory, dt: float, d_min: float) -> dict:
    """
    Compute evaluation metrics for a trajectory.

    Returns dict with:
    - rho_min: minimum robustness
    - satisfaction_rate: fraction with rho > 0
    - collisions: number of time steps with d < 0
    - takeovers: number of large authority changes (|delta_alpha| > 0.2)
    - monotonicity_violations: number of emergency monotonicity violations
    - total_variation: sum |alpha_{k+1}-alpha_k|
    - jerk_max: maximum jerk [m/s^3]
    - distance_min: minimum distance [m]
    - ttc_min: minimum TTC [s]
    - alpha_min/max: authority range
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")

    distance = np.asarray(traj.distance, dtype=float)
    ttc = np.asarray(traj.ttc, dtype=float)
    alpha = np.asarray(traj.alpha, dtype=float)
    emergency = np.asarray(traj.emergency, dtype=bool)
    a_ego = np.asarray(traj.a_ego, dtype=float)

    rho_min = float(compute_rho_min(distance.tolist(), d_min))
    satisfaction_rate = float(compute_satisfaction_rate(distance.tolist(), d_min))

    collisions = int(np.sum(distance < 0.0))

    dalpha = np.diff(alpha)
    takeovers = int(np.sum(np.abs(dalpha) > 0.2))

    violations = int(np.sum(emergency[:-1] & (alpha[1:] < alpha[:-1])))

    tv = float(np.sum(np.abs(dalpha)))

    if len(a_ego) >= 2:
        jerk = np.abs(np.diff(a_ego) / float(dt))
        jerk_max = float(np.max(jerk))
    else:
        jerk_max = 0.0

    return {
        "rho_min": rho_min,
        "satisfaction_rate": satisfaction_rate,
        "collisions": collisions,
        "takeovers": takeovers,
        "monotonicity_violations": violations,
        "total_variation": tv,
        "jerk_max": jerk_max,
        "distance_min": float(np.min(distance)) if distance.size else float("nan"),
        "ttc_min": float(np.min(ttc)) if ttc.size else float("nan"),
        "alpha_min": float(np.min(alpha)) if alpha.size else float("nan"),
        "alpha_max": float(np.max(alpha)) if alpha.size else float("nan"),
    }


def save_trajectory_csv(traj: Trajectory, output_path: Path):
    """Save trajectory data to CSV."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t", "x_ego", "v_ego", "a_ego",
            "x_lead", "v_lead", "a_lead",
            "distance", "ttc", "emergency", "robustness",
            "alpha", "u_auto", "u_human", "u_blend",
        ])
        for i in range(len(traj.t)):
            writer.writerow([
                traj.t[i], traj.x_ego[i], traj.v_ego[i], traj.a_ego[i],
                traj.x_lead[i], traj.v_lead[i], traj.a_lead[i],
                traj.distance[i], traj.ttc[i], int(traj.emergency[i]), traj.robustness[i],
                traj.alpha[i], traj.u_auto[i], traj.u_human[i], traj.u_blend[i],
            ])

    print(f"Trajectory saved to {output_path}")


def _print_header(title: str):
    print("=" * 60)
    print(title)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Neuro-Symbolic Authority Allocation POC")

    # core run controls
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility (default: 0)")
    parser.add_argument("--outdir", type=str, default="output",
                        help="Output directory (default: output)")

    # scenario controls
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Sampling period [s] (default: 0.1)")
    parser.add_argument("--horizon", type=float, default=30.0,
                        help="Simulation horizon [s] (default: 30.0)")
    parser.add_argument("--d-min", type=float, default=10.0,
                        help="Minimum safe distance d_min [m] (default: 10.0)")
    parser.add_argument("--tau-emerg", type=float, default=2.0,
                        help="Emergency TTC threshold [s] (default: 2.0)")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Max authority rate gamma [1/s] (default: 0.5)")
    parser.add_argument("--eta", type=float, default=0.3,
                        help="Proposal inertia eta (default: 0.3)")
    parser.add_argument("--sigma-w", type=float, default=0.10,
                        help="Ego process noise std sigma_w [m/s^2] (default: 0.10)")
    parser.add_argument("--alpha0", type=float, default=0.2,
                        help="Initial/nominal authority alpha0 (default: 0.2)")

    # output controls
    parser.add_argument("--ttc-max", type=float, default=10.0,
                        help="TTC plot cap [s] (default: 10.0)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Figure DPI (default: 150)")
    parser.add_argument("--no-csv", action="store_true",
                        help="Disable saving CSV time-series outputs")
    parser.add_argument("--no-fig", action="store_true",
                        help="Disable saving the figure output")

    # LLM controls
    parser.add_argument("--llm-backend", type=str, default="mock",
                        choices=["mock", "openai", "hf"],
                        help="LLM backend: mock | openai | hf (default: mock)")
    parser.add_argument("--llm-model", type=str, default="",
                        help="Model name/id. For openai: gpt-5-nano. For hf: meta-llama/Llama-3.1-8B, google/gemma-2-9b-it")
    parser.add_argument("--llm-period", "--llm-period-s", dest="llm_period", type=float, default=1.0,
                        help="Call the LLM at most once per this many seconds (default: 1.0)")
    parser.add_argument("--llm-verbose", action="store_true",
                        help="Enable verbose logging inside LLM backend (confirms calls)")
    parser.add_argument("--real-llm", action="store_true",
                        help="(Deprecated) Same as --llm-backend openai")

    args = parser.parse_args()

    if args.dt <= 0:
        raise SystemExit("Error: --dt must be > 0")
    if args.horizon <= 0:
        raise SystemExit("Error: --horizon must be > 0")
    if args.d_min < 0:
        raise SystemExit("Error: --d-min must be >= 0")
    if args.gamma < 0:
        raise SystemExit("Error: --gamma must be >= 0")
    if not (0.0 <= args.eta <= 1.0):
        raise SystemExit("Error: --eta must be in [0, 1]")
    if args.llm_period <= 0:
        raise SystemExit("Error: --llm-period must be > 0")
    if not (0.0 <= args.alpha0 <= 1.0):
        raise SystemExit("Error: --alpha0 must be in [0, 1]")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Backward compatibility: --real-llm implies OpenAI backend
    llm_backend = args.llm_backend
    if args.real_llm:
        llm_backend = "openai"

    llm_model = args.llm_model.strip()
    if llm_backend == "openai" and not llm_model:
        llm_model = "gpt-5-nano"

    if llm_backend == "hf" and not llm_model:
        raise SystemExit("Error: --llm-backend hf requires --llm-model (e.g., meta-llama/Llama-3.1-8B)")

    # setup scenario parameters
    params = ScenarioParams(
        dt=args.dt,
        T=args.horizon,
        x0_ego=0.0,
        v0_ego=30.0,
        x0_lead=30.0,
        v0_lead=30.0,
        alpha0=float(args.alpha0),
        d_min=args.d_min,
        tau_emerg=args.tau_emerg,
        gamma=args.gamma,
        eta=args.eta,
        sigma_w=args.sigma_w,
        llm_backend=llm_backend,
        llm_model=llm_model if llm_model else "gpt-5-nano",
        llm_period_s=float(args.llm_period),
        llm_verbose=bool(args.llm_verbose),
    )

    _print_header("Neuro-Symbolic Authority Allocation POC")

    print(f"Python: {sys.version.split()[0]}")
    print(f"Seed: {args.seed}")
    print(f"Horizon: {params.T}s, dt: {params.dt}s")
    print(f"d_min: {params.d_min}m, tau_emerg: {params.tau_emerg}s, gamma: {params.gamma}/s, eta: {params.eta}")
    print(f"Noise sigma_w: {params.sigma_w}")
    print(f"alpha0: {params.alpha0}")
    print(f"Output dir: {outdir.resolve()}")
    print()

    if llm_backend == "mock":
        print("LLM backend: MOCK (rule-based MockLLM)")
    elif llm_backend == "openai":
        has_key = bool(os.getenv("OPENAI_API_KEY"))
        print(f"LLM backend: OPENAI (model={llm_model})")
        print(f"OPENAI_API_KEY present: {has_key}")
        if not has_key:
            print("WARNING: OPENAI_API_KEY not set. RealLLM will fall back to safe HOLD at runtime.")
    else:
        print(f"LLM backend: OFFLINE_HF (model_id={llm_model})")
        print("NOTE: This requires the model to be present in the local Hugging Face cache.")
    print(f"LLM period: {params.llm_period_s:.3f}s")
    print(f"LLM verbose: {bool(params.llm_verbose)}")
    print()

    print("Running simulations...")
    t0 = time.time()
    results: Dict[str, Trajectory] = run_all_baselines(seed=args.seed, params=params)
    t1 = time.time()
    print(f"✓ Simulations complete in {t1 - t0:.3f}s")
    print()

    if not args.no_fig:
        print("Generating four-panel figure...")
        fig = plot_four_panel_figure(
            results,
            gamma=params.gamma,
            T=params.T,
            d_min=params.d_min,
            ttc_max=args.ttc_max,
        )
        fig_path = outdir / "figure.png"
        save_figure(fig, fig_path, dpi=int(args.dpi))
        print()

    print("Computing metrics...")
    all_metrics = {}
    for name, traj in results.items():
        metrics = compute_metrics(traj, params.dt, params.d_min)
        all_metrics[name] = metrics

        print(f"\n{name.upper()}:")
        print(f"  ρ_min:               {metrics['rho_min']:>8.3f} m")
        print(f"  Satisfaction rate:   {metrics['satisfaction_rate']:>8.2%}")
        print(f"  Collisions:          {metrics['collisions']:>8d}")
        print(f"  Takeovers:           {metrics['takeovers']:>8d}")
        print(f"  Mono. violations:    {metrics['monotonicity_violations']:>8d}")
        print(f"  Total variation:     {metrics['total_variation']:>8.3f}")
        print(f"  Max jerk:            {metrics['jerk_max']:>8.3f} m/s³")
        print(f"  Min distance:        {metrics['distance_min']:>8.3f} m")
        print(f"  Min TTC:             {metrics['ttc_min']:>8.3f} s")
        print(f"  α range:             [{metrics['alpha_min']:.3f}, {metrics['alpha_max']:.3f}]")

    metrics_path = outdir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    print()

    if not args.no_csv:
        print("Saving time-series data...")
        for name, traj in results.items():
            csv_path = outdir / f"timeseries_{name}.csv"
            save_trajectory_csv(traj, csv_path)
        print()

    _print_header("POC complete!")


if __name__ == "__main__":
    main()
