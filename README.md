# 🧠🤝 Neuro-Symbolic Authority Allocation

## 🧾 Overview
This repository contains a research implementation of neuro-symbolic authority allocation for shared control in a longitudinal car-following scenario with an emergency braking event. At each discrete simulation step, an authority variable **$\alpha_k \in [0,1]$** blends automation and human control inputs, enabling dynamic shifts of control priority as safety risk changes. The code evaluates three strategies: **(i)** a Classical baseline that updates authority using hand-crafted risk thresholds, **(ii)** an LLM-only baseline that converts intent recommendations into authority updates without a formal safety monitor (but with a nominal authority floor to avoid collapse under stochastic human input), and **(iii)** a Neuro-Symbolic (NeSy) method that applies an invariant projection layer to enforce constraints such as bounded rate of change, emergency/violation non-decreasing authority, and safety-aware authority floors. LLM calls are optionally throttled via `--llm-period`, reusing the last intent between calls. The simulator logs full time-series trajectories (states, safety signals, control actions, and authority) and aggregates quantitative metrics for comparison across methods. Runs are deterministic for a fixed seed to support consistent evaluation and repeatable experiments.


## 🏗️ System Architecture

<div align="center">
  <img src="data/image/authority_allocation_architecture.png" alt="Authority Allocation System Architecture" width="800"/>
</div>

The diagram above illustrates the neuro-symbolic authority allocation framework, showing how LLM-generated intents are processed through an invariant projection layer to ensure safety constraints while dynamically adjusting authority between automation and human control.

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

From the project root (the folder that contains `main.py`), run:
```bash
python main.py --seed 0 --outdir output
```

Runs all three baselines (`classical`, `llm_only`, `nesy`) with deterministic seeding and writes results to `output/`.

### Core Usage Patterns

**1) Deterministic experiments with different seeds**
```bash
python main.py --seed 42 --outdir output
```

**2) OpenAI backend (requires OPENAI_API_KEY)**
```bash
export OPENAI_API_KEY="YOUR_KEY"
python main.py --llm-backend openai --llm-model gpt-4.1-mini --outdir output
```

**3) Stress test (adversarial LLM + repeated emergencies)**
```bash
python main.py --stress-test --outdir output
# or manually configure:
python main.py --llm-backend adversarial --llm-attack-mode force_decrease --lead-scenario repeated --outdir output
```

**4) Tune authority floors for fair baseline comparison**
```bash
# Conservative: higher automation authority across all baselines
python main.py --alpha0 0.3 --llm-only-alpha-min 0.3 --nesy-floor-emergency 0.90 --outdir output
```

**5) Adjust safety parameters**
```bash
python main.py --d-min 12 --tau-emerg 1.5 --gamma 0.4 --eta 0.25 --outdir output
```

### Advanced Usage

**Lead vehicle scenarios:**
- `--lead-scenario standard`: Single moderate braking event (10-14s)
- `--lead-scenario severe`: Extended braking duration (10-15s)
- `--lead-scenario repeated`: Multiple emergency onsets (8-11s, 16-19s, 24-27s)

**Adversarial backend (stress-tests monitor robustness):**
```bash
python main.py --llm-backend adversarial --llm-attack-mode <MODE> --outdir output
# MODE: oscillate | force_decrease | random_extreme | confidence_manipulation
```

**Authority floor parameters:**
- `--alpha0 <float>`: Initial authority and nominal floor (default: 0.2)
- `--llm-only-alpha-min <float>`: LLM-only baseline floor (default: 0.0)
- `--nesy-floor-emergency <float>`: NeSy floor during emergency (default: 0.85)
- `--nesy-floor-violation <float>`: NeSy floor during safety violation (default: 0.70)

**Offline Hugging Face (must be cached locally):**
```bash
python main.py --llm-backend hf --llm-model meta-llama/Llama-3.1-8B --outdir output
```

**Reduce LLM calls (cost/latency control):**
```bash
python main.py --llm-backend openai --llm-model gpt-4.1-mini --llm-period 2.0 --outdir output
```

**Output controls:**
- `--no-fig`: Disable figure generation
- `--no-csv`: Disable CSV time-series exports
- `--dpi <int>`: Figure resolution (default: 150)
- `--ttc-max <float>`: TTC plot cap in seconds (default: 10.0)

### Troubleshooting

- **If Python can't import src.***, run from the repo root (check that `main.py` and `src/` are present): `ls main.py src`
- **If OpenAI mode feels "stuck"**, increase `--llm-period` (fewer calls), or use mock/offline HF
- **If LLM-based baselines look unsafe**, raise `--alpha0` (keeps a nominal authority floor), and/or reduce `--llm-period` jitter by calling less often
- **If offline HF runs slow**, your machine is likely CPU/offloading; try 4-bit quantization support (`bitsandbytes`) or use a GPU

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

## 📧 Contact
For questions or issues: safayat.b.hakim@gmail.com

## 📝 Notes
This codebase is intended as a compact research implementation aligned with an accompanying paper.
