# mc_tuner — Automatic JAX Hyperparameter Tuner

A framework that automatically tunes JAX execution-level parameters (scan unrolling, buffer donation, numerical precision, activation checkpointing, and more) for arbitrary user-defined JAX functions. Built as a diploma thesis project.

---

## What it does

JAX exposes a set of low-level knobs that control how code is compiled and executed — things like `lax.scan` unroll depth, `jit` buffer donation, and matmul precision. The right values depend on the hardware, the algorithm, and the batch size. Picking them by hand is tedious and non-obvious.

`mc_tuner` instruments your function at the source level (via LibCST), injects these knobs as tunable parameters, and searches over the resulting space using bandit or Bayesian optimisation agents. It returns the fastest configuration that stays within a user-specified numerical precision budget.

---

## Tunable parameters

| Parameter | Values | Effect |
|---|---|---|
| `scan_unroll` | 1, 2, 4, 8, 16 | Loop unrolling depth inside `lax.scan` |
| `scan_reverse` | False, True | Traversal direction in `lax.scan` |
| `jit_donate_argnums` | (), (0,), (1,), (0,1) | Buffer donation (reuse input memory for output) |
| `dot_precision` | None, "high", "highest" | Floating-point precision for `jnp.dot` |
| `matmul_precision` | None, "high", "highest" | Floating-point precision for `jnp.matmul` |
| `autotune_level` | 0–4 | XLA kernel autotuning exhaustiveness |
| `map_chunk_size` | 1, 4, 8, 16, None | Chunk size for `lax.map` |
| `checkpoint_policy` | None + 3 JAX policies | Activation checkpointing strategy |

---

## Installation

**Requirements:** Python 3.10+, CUDA 12 (for GPU support)

```bash
git clone <repo>
cd mc_tuner
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For CPU-only (no CUDA):
```bash
pip install jax jaxlib libcst==1.8.6 optuna==4.8.0 numpy==1.26.4 scipy==1.15.3 matplotlib==3.10.8
```

---

## Quick start — AutoTuner

`AutoTuner` works on any JAX function defined in a `.py` file. It instruments the source automatically.

```python
import jax
import jax.numpy as jnp
from mc_tuner import AutoTuner

def simulate(key, n_steps=500):
    prices = jax.random.normal(key, (n_steps,))
    def step(carry, x):
        return carry + x, carry + x
    _, out = jax.lax.scan(step, 0.0, prices)
    return jnp.mean(out)

def generate_inputs(master_key, batch_size):
    return jax.random.split(master_key, batch_size)

tuner = AutoTuner.from_fn(
    fn=simulate,
    generate_inputs=generate_inputs,
    precision=0.01,   # max allowed output drift from baseline
    n_repeats=5,
)

best = tuner.run(method="grid", batch_size=64, verbose=True)
tuner.compare(batch_size=64)
print(best)  # {'scan_unroll': 8, 'scan_reverse': False, ...}
```

### Search methods

| Method | Description |
|---|---|
| `"grid"` | Exhaustive search over all configs |
| `"random"` | Uniform random sampling (`n` configs) |
| `"rl"` | Bandit agent (see `agent=` parameter) |

### RL agents

```python
best = tuner.run(method="rl", agent="ucb", episodes=200)
# agent options: "ucb", "epsilon_greedy", "softmax", "thompson"
```

---

## Project structure

```
mc_tuner/
├── auto_tuner.py       # AutoTuner: end-to-end instrument → search → compare
├── instrumentor.py     # LibCST-based source instrumentation
├── search_space.py     # JAX_INJECT_SPACE definitions and Cartesian product helper
├── evaluator.py        # Old-style Evaluator (step_fn + init_carry_fn interface)
├── scoring.py          # Precision modes, drift metrics, reward function
├── simulation.py       # JIT-compiled vmap simulation engine (old pipeline)
├── tuner.py            # Old-style Tuner (wraps Evaluator + Searcher)
├── io.py               # JSON serialisation for configs and results
├── hardware.py         # XLA flag space builder
├── agents/
│   ├── bandit.py       # Epsilon-greedy
│   ├── ucb.py          # UCB1
│   ├── thompson.py     # Thompson sampling
│   ├── softmax.py      # Boltzmann / softmax bandit
│   ├── sac.py          # Soft Actor-Critic (pure JAX neural net)
│   ├── grpo.py         # GRPO policy gradient agent
│   └── autosampler.py  # Optuna TPE wrapper
├── searchers/
│   ├── grid_search.py
│   ├── random_search.py
│   └── rl_search.py
└── tests/
    ├── test_unroll.py          # scan_unroll on GBM path simulation
    ├── test_reverse.py         # scan_reverse on discounted RL returns
    ├── test_precision.py       # dot_precision on GP posterior mean
    ├── test_chunk_size.py      # map_chunk_size
    ├── test_checkpoint.py      # checkpoint_policy
    ├── test_donate_argnums.py  # jit_donate_argnums on HMC
    └── bench_scan_reverse_cpu.py  # CPU memory-access benchmark

experiments/
├── domains/            # Reusable algorithm implementations
│   ├── american_option.py
│   ├── european_option.py
│   ├── basket_option.py
│   ├── runge_kutta.py
│   ├── kalman_smoother.py
│   └── kmc_random_walk.py
├── buffer_donation/    # Throughput vs donation sweep (state_dim × donate config)
├── convergence_speed/  # Agent convergence curves comparison
├── distributional_integrity/  # Wasserstein drift validation
├── precision_tradeoff/ # Pareto frontier: speedup vs numerical accuracy
├── speedup_magnitude/  # Wall-time speedup across 5 diverse domains
└── tuning_roi/         # Break-even analysis: when does tuning pay off?
```

---

## Running experiments

Each experiment is self-contained:

```bash
# Speedup across domains
python experiments/speedup_magnitude/run.py

# Agent convergence comparison
python experiments/convergence_speed/run.py

# Buffer donation sweep
python experiments/buffer_donation/run.py

# Precision-speedup Pareto frontier
python experiments/precision_tradeoff/run.py
```

Results are saved to `experiments/<name>/results/results.json` and figures to `experiments/<name>/figures/`.

---

## Running tests

```bash
python mc_tuner/tests/test_unroll.py
python mc_tuner/tests/test_precision.py
python mc_tuner/tests/test_reverse.py
python mc_tuner/tests/test_donate_argnums.py
python mc_tuner/tests/bench_scan_reverse_cpu.py   # forces CPU backend
```

---

## Precision modes

The `precision` argument controls how much numerical drift from the baseline is acceptable. The mode is set via `precision_mode` (default `"absolute"`):

| Mode | Meaning |
|---|---|
| `"absolute"` | `|mean(candidate) - mean(baseline)| ≤ threshold` |
| `"relative"` | Relative deviation as a fraction of baseline |
| `"zscore"` | Deviation in units of baseline standard deviation |
| `"wasserstein"` | Wasserstein-1 distance between output distributions |
| `"quantile"` | Max quantile drift across 7 probability levels |
| `"soft"` | Gaussian-penalised reward (no hard gate) |

---

## How instrumentation works

`AutoTuner.from_fn` retrieves the source of your function with `inspect.getsource`, parses it with LibCST, and rewrites call sites:

```python
# Original
jax.lax.scan(step, init, xs)

# Instrumented
jax.lax.scan(step, init, xs, unroll=scan_unroll, reverse=scan_reverse)
```

The rewritten function is compiled with `exec()` in a sandbox that shares the original function's globals. Parameters are injected as keyword arguments at call time — no monkey-patching, no import hooks.
