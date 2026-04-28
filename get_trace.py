# profile_compare.py
import jax
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mc_tuner.tuner import Tuner
from mc_tuner.search_space import JAX_INJECT_DEFAULTS
from experiments.domains.american_option import make_fns as option_fns

step_fn, init_carry_fn = option_fns(n_steps=252)

tuner = Tuner(
    step_fn       = step_fn,
    init_carry_fn = init_carry_fn,
    master_key    = jax.random.PRNGKey(0),
    batch_size    = 512,
    n_steps       = 252,
)

# 1. Trace baseline (unoptimized)
tuner.trace(JAX_INJECT_DEFAULTS, output_dir="./traces/baseline")

best_params, _ = tuner.run(method="rl", agent="autosampler", episodes=60, verbose=True)
tuner.trace(best_params, output_dir="./traces/tuned")

print("Done. Open both in https://ui.perfetto.dev")
