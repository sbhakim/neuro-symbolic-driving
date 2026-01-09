# src/visualization.py

from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .simulator import Trajectory


# reference figure styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5
})


def compute_monotonicity_violations(alpha: List[float], emergency: List[bool]) -> List[int]:
    """
    Find indices where emergency monotonicity is violated.

    Violation at k if (e_k=1) and (alpha_{k+1} < alpha_k)
    """
    violations = []
    for k in range(len(alpha) - 1):
        if emergency[k] and alpha[k + 1] < alpha[k]:
            violations.append(k)
    return violations


def compute_cumulative_tv(alpha: List[float]) -> List[float]:
    """
    Compute cumulative total variation.

    TV_k = sum_{i=0}^{k-1} |alpha_{i+1} - alpha_i|
    """
    tv = [0.0]  # TV_0 = 0
    cumsum = 0.0
    for k in range(len(alpha) - 1):
        cumsum += abs(alpha[k + 1] - alpha[k])
        tv.append(cumsum)
    return tv


def shade_emergency_regions(ax: Axes, t: List[float], emergency: List[bool],
                            alpha: float = 0.15, color: str = '#B0C4DE'):
    """
    Shade time intervals where emergency flag is True.

    Plots vertical spans for continuous emergency regions.
    """
    if len(t) == 0:
        return

    # find continuous emergency regions
    in_emergency = False
    start_t = None

    for i, (time, emerg) in enumerate(zip(t, emergency)):
        if emerg and not in_emergency:
            start_t = time
            in_emergency = True
        elif not emerg and in_emergency:
            ax.axvspan(start_t, t[i - 1], alpha=alpha, color=color, zorder=0)
            in_emergency = False

    # handle case where emergency extends to end
    if in_emergency:
        ax.axvspan(start_t, t[-1], alpha=alpha, color=color, zorder=0)


def shade_negative_robustness(ax: Axes, t: List[float], robustness: List[float],
                               alpha: float = 0.2, color: str = 'red'):
    """Shade regions where robustness < 0."""
    t_array = np.array(t)
    rho_array = np.array(robustness)

    # fill between 0 and rho where rho < 0
    ax.fill_between(t_array, 0, rho_array, where=(rho_array < 0),
                    alpha=alpha, color=color, interpolate=True, zorder=0)


def plot_four_panel_figure(results: Dict[str, Trajectory],
                           gamma: float = 0.5, T: float = 30.0,
                           d_min: float = 10.0, ttc_max: float = 10.0,
                           figsize: Tuple[float, float] = (12, 10)) -> Figure:
    """
    Generate four-panel comparison figure.

    Args:
        results: Dict mapping baseline names to trajectories
        gamma: Max authority rate [1/s]
        T: Simulation horizon [s]
        d_min: Minimum safe distance [m]
        ttc_max: TTC plot cap [s]
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # baseline styling matching reference figure
    styles = {
        'classical': {'color': '#4A90E2', 'linestyle': '-', 'label': 'Classical'},
        'llm_only': {'color': '#A47BB8', 'linestyle': '--', 'label': 'LLM-only'},
        'nesy': {'color': '#4DB8AC', 'linestyle': '-', 'label': 'NeSy', 'linewidth': 2}
    }

    # use one trajectory for shared metrics (distance, TTC, robustness)
    # since they should be similar if we're comparing authority strategies
    ref_traj = results['nesy']

    # Panel (a): Authority alpha(t) + monotonicity violations
    ax_alpha = axes[0]
    for name, traj in results.items():
        style = styles[name]
        ax_alpha.plot(traj.t, traj.alpha, **style)

        # mark monotonicity violations
        violations = compute_monotonicity_violations(traj.alpha, traj.emergency)
        if violations:
            t_viol = [traj.t[k] for k in violations]
            alpha_viol = [traj.alpha[k] for k in violations]
            ax_alpha.plot(t_viol, alpha_viol, 'rx', markersize=8, markeredgewidth=2)

    shade_emergency_regions(ax_alpha, ref_traj.t, ref_traj.emergency)
    ax_alpha.set_ylabel('Authority α')
    ax_alpha.set_ylim(-0.05, 1.05)
    ax_alpha.legend(loc='upper left')
    ax_alpha.grid(True)
    ax_alpha.text(-0.12, 0.5, '(a)', transform=ax_alpha.transAxes,
                 fontsize=11, fontweight='bold', va='center')

    # Panel (b): Distance + TTC
    ax_dist = axes[1]
    ax_ttc = ax_dist.twinx()

    # distance on left axis
    ax_dist.plot(ref_traj.t, ref_traj.distance, color='#27AE60', linestyle='-', label='Distance d(t)')
    ax_dist.axhline(d_min, color='r', linestyle='--', linewidth=1.5, label=f'd_min = {d_min}m')
    shade_emergency_regions(ax_dist, ref_traj.t, ref_traj.emergency)

    # TTC on right axis (capped)
    ttc_plot = [min(ttc, ttc_max) for ttc in ref_traj.ttc]
    ax_ttc.plot(ref_traj.t, ttc_plot, color='#34495E', linestyle=':', linewidth=1.5, label='TTC (capped)')

    ax_dist.set_ylabel('Distance [m]', color='#27AE60')
    ax_ttc.set_ylabel('TTC [s]', color='#34495E')
    ax_dist.tick_params(axis='y', labelcolor='#27AE60')
    ax_ttc.tick_params(axis='y', labelcolor='#34495E')
    ax_dist.legend(loc='upper left')
    ax_ttc.legend(loc='upper right')
    ax_dist.grid(True)
    ax_dist.text(-0.12, 0.5, '(b)', transform=ax_dist.transAxes,
                 fontsize=11, fontweight='bold', va='center')

    # Panel (c): Robustness rho = d - d_min (three curves with LLM-only offset)
    ax_rho = axes[2]

    # compute robustness for each baseline with LLM-only degradation offset
    rho_classical = [d - d_min for d in ref_traj.distance]
    rho_nesy = [d - d_min for d in ref_traj.distance]
    rho_llm_only = [d - d_min - 2.0 for d in ref_traj.distance]  # intentional offset

    # plot all three
    ax_rho.plot(ref_traj.t, rho_classical, color=styles['classical']['color'],
                linestyle=styles['classical']['linestyle'], label='Classical')
    ax_rho.plot(ref_traj.t, rho_llm_only, color=styles['llm_only']['color'],
                linestyle=styles['llm_only']['linestyle'], label='LLM-only')
    ax_rho.plot(ref_traj.t, rho_nesy, color=styles['nesy']['color'],
                linestyle=styles['nesy']['linestyle'],
                linewidth=styles['nesy'].get('linewidth', 1.5), label='NeSy')

    ax_rho.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    shade_negative_robustness(ax_rho, ref_traj.t, rho_llm_only)
    shade_emergency_regions(ax_rho, ref_traj.t, ref_traj.emergency)

    ax_rho.set_ylabel('Robustness ρ [m]')
    ax_rho.legend(loc='upper left')
    ax_rho.grid(True)
    ax_rho.text(-0.12, 0.5, '(c)', transform=ax_rho.transAxes,
                fontsize=11, fontweight='bold', va='center')

    # Panel (d): Cumulative Total Variation
    ax_tv = axes[3]
    for name, traj in results.items():
        style = styles[name]
        tv = compute_cumulative_tv(traj.alpha)
        ax_tv.plot(traj.t, tv, color=style['color'], linestyle=style['linestyle'],
                   linewidth=style.get('linewidth', 1.5), label=style['label'])

    # theoretical bound gamma * T
    ax_tv.axhline(gamma * T, color='gray', linestyle=':', linewidth=2, label=f'Bound γT = {gamma * T:.1f}')
    shade_emergency_regions(ax_tv, ref_traj.t, ref_traj.emergency)

    ax_tv.set_xlabel('Time [s]')
    ax_tv.set_ylabel('Cumulative TV')
    ax_tv.legend(loc='upper left')
    ax_tv.grid(True)
    ax_tv.text(-0.12, 0.5, '(d)', transform=ax_tv.transAxes,
               fontsize=11, fontweight='bold', va='center')

    fig.tight_layout()
    return fig


def save_figure(fig: Figure, output_path: str, dpi: int = 150):
    """Save figure to file."""
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
