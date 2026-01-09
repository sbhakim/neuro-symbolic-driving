# main.py

import argparse
import json
from pathlib import Path

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
    - total_variation: final TV_N
    - jerk_max: maximum jerk [m/s^3]
    """
    # robustness metrics
    rho_min = compute_rho_min(traj.distance, d_min)
    satisfaction_rate = compute_satisfaction_rate(traj.distance, d_min)

    # collision count
    collisions = sum(1 for d in traj.distance if d < 0)

    # takeover count
    takeovers = 0
    for k in range(len(traj.alpha) - 1):
        if abs(traj.alpha[k + 1] - traj.alpha[k]) > 0.2:
            takeovers += 1

    # monotonicity violations
    violations = 0
    for k in range(len(traj.alpha) - 1):
        if traj.emergency[k] and traj.alpha[k + 1] < traj.alpha[k]:
            violations += 1

    # total variation
    tv = sum(abs(traj.alpha[k + 1] - traj.alpha[k]) for k in range(len(traj.alpha) - 1))

    # jerk (|da/dt|)
    jerk_max = 0.0
    for k in range(1, len(traj.a_ego)):
        jerk = abs((traj.a_ego[k] - traj.a_ego[k - 1]) / dt)
        jerk_max = max(jerk_max, jerk)

    return {
        'rho_min': float(rho_min),
        'satisfaction_rate': float(satisfaction_rate),
        'collisions': int(collisions),
        'takeovers': int(takeovers),
        'monotonicity_violations': int(violations),
        'total_variation': float(tv),
        'jerk_max': float(jerk_max)
    }


def save_trajectory_csv(traj: Trajectory, output_path: Path):
    """Save trajectory data to CSV."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # header
        writer.writerow([
            't', 'x_ego', 'v_ego', 'a_ego',
            'x_lead', 'v_lead', 'a_lead',
            'distance', 'ttc', 'emergency', 'robustness',
            'alpha', 'u_auto', 'u_human', 'u_blend'
        ])
        # data
        for i in range(len(traj.t)):
            writer.writerow([
                traj.t[i], traj.x_ego[i], traj.v_ego[i], traj.a_ego[i],
                traj.x_lead[i], traj.v_lead[i], traj.a_lead[i],
                traj.distance[i], traj.ttc[i], int(traj.emergency[i]), traj.robustness[i],
                traj.alpha[i], traj.u_auto[i], traj.u_human[i], traj.u_blend[i]
            ])

    print(f"Trajectory saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Neuro-Symbolic Authority Allocation POC'
    )
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility (default: 0)')
    parser.add_argument('--outdir', type=str, default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--dt', type=float, default=0.1,
                       help='Sampling period [s] (default: 0.1)')
    parser.add_argument('--horizon', type=float, default=30.0,
                       help='Simulation horizon [s] (default: 30.0)')

    args = parser.parse_args()

    # create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # setup scenario parameters
    params = ScenarioParams(
        dt=args.dt,
        T=args.horizon,
        x0_ego=0.0,
        v0_ego=30.0,
        x0_lead=30.0,
        v0_lead=30.0,
        alpha0=0.2,
        d_min=10.0,
        tau_emerg=2.0,
        gamma=0.5,
        eta=0.3,
        sigma_w=0.10
    )

    print("=" * 60)
    print("Neuro-Symbolic Authority Allocation POC")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Horizon: {params.T}s, dt: {params.dt}s")
    print(f"Output: {outdir}")
    print()

    # run all baselines
    print("Running simulations...")
    results = run_all_baselines(seed=args.seed, params=params)
    print("✓ Simulations complete")
    print()

    # generate figure
    print("Generating four-panel figure...")
    fig = plot_four_panel_figure(
        results,
        gamma=params.gamma,
        T=params.T,
        d_min=params.d_min,
        ttc_max=10.0
    )
    save_figure(fig, outdir / 'figure.png', dpi=150)
    print()

    # compute and save metrics
    print("Computing metrics...")
    all_metrics = {}
    for name, traj in results.items():
        metrics = compute_metrics(traj, params.dt, params.d_min)
        all_metrics[name] = metrics

        print(f"\n{name.upper()}:")
        print(f"  ρ_min:             {metrics['rho_min']:>8.3f} m")
        print(f"  Satisfaction rate: {metrics['satisfaction_rate']:>8.2%}")
        print(f"  Collisions:        {metrics['collisions']:>8d}")
        print(f"  Takeovers:         {metrics['takeovers']:>8d}")
        print(f"  Mono. violations:  {metrics['monotonicity_violations']:>8d}")
        print(f"  Total variation:   {metrics['total_variation']:>8.3f}")
        print(f"  Max jerk:          {metrics['jerk_max']:>8.3f} m/s³")

    # save metrics to JSON
    metrics_path = outdir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    print()

    # save time-series data
    print("Saving time-series data...")
    for name, traj in results.items():
        csv_path = outdir / f'timeseries_{name}.csv'
        save_trajectory_csv(traj, csv_path)
    print()

    print("=" * 60)
    print("POC complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
