# 🧠🤝 Neuro-Symbolic Authority Allocation

## 🧾 Overview
This repository contains a research implementation of neuro-symbolic authority allocation for shared control in a longitudinal car-following scenario with an emergency braking event. At each discrete time step, an authority variable **$\alpha_k \in [0,1]$** blends automation and human control inputs, enabling dynamic shifts of control priority as safety risk changes. The code evaluates three strategies: **(i)** a Classical baseline that updates authority using hand-crafted risk thresholds, **(ii)** an LLM-only baseline that converts intent recommendations into authority updates without formal safety enforcement, and **(iii)** a Neuro-Symbolic (NeSy) method that applies an invariant projection layer to enforce constraints such as bounded rate of change and emergency monotonicity. The simulator logs full time-series trajectories (states, safety signals, control actions, and authority) and aggregates quantitative metrics for comparison across methods. Runs are deterministic for a fixed seed to support consistent evaluation and repeatable experiments.

## 🗂️ Repository Structure
- `main.py` — Runs the simulation suite and writes outputs  
- `src/vehicle.py` — Vehicle state, discrete-time dynamics, safety metric computation  
- `src/controllers.py` — Automation controller (PD car-following) and human controller (delay + noise)  
- `src/stl_monitor.py` — Robustness proxy utilities and aggregate statistics  
- `src/authority_allocator.py` — Authority allocators (Classical / LLM-only / NeSy) and invariant projection  
- `src/llm_interface.py` — `AuthorityIntent` schema and mock intent generator  
- `src/simulator.py` — Scenario runner and time-series trajectory logging  
- `src/visualization.py` — Plot generation and export  
- `tests/` — Unit tests for dynamics, safety metrics, and monitoring utilities  

## 📦 Dependencies
Key packages:
- Python (3.9 recommended)
- `numpy`
- `matplotlib`
- `pytest`

Optional:
- `rtamt` (only if extending the monitoring layer beyond the current robustness proxy)

## 🏁 Running
Example run (writes to `output/`):
```bash
python main.py --seed 0 --outdir output
```

## 📤 Outputs

The program writes artifacts to the selected output directory (default: `output/`):

- `metrics.json` — Per-baseline summary metrics with keys:
  - `rho_min`
  - `satisfaction_rate`
  - `collisions`
  - `takeovers`
  - `monotonicity_violations`
  - `total_variation`
  - `jerk_max`

- `timeseries_classical.csv`
- `timeseries_llm_only.csv`
- `timeseries_nesy.csv`

Each CSV contains:  
`t, x_ego, v_ego, a_ego, x_lead, v_lead, a_lead, distance, ttc, emergency, robustness, alpha, u_auto, u_human, u_blend`

- `figure.png` — A saved visualization artifact (PNG)

All outputs are deterministic for a fixed seed.

## 🧪 Testing

- `tests/test_vehicle.py` — Vehicle dynamics, TTC/emergency logic, lead profile behavior  
- `tests/test_monitor.py` — Robustness and aggregate monitoring utilities  

Run tests:
```bash
pytest -q
```

## Contact
For questions or issues: safayat.b.hakim@gmail.com

## Notes
This codebase is intended as a compact research implementation aligned with an accompanying paper.
