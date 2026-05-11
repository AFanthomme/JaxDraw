'''
Tests already executed the cumulated rewards part, but sanity checks are too long so keep them separate
'''
from pathlib import Path
from src.config import EnvParams, RulesetLiteral
from src.baseline_policies import baseline_policy_register
import jax.numpy as jnp
from pathlib import Path
from src.eval_utils import sanity_check, compute_running_statistics
from src.image_utils import plot_cumulated_rewards
from src.custom_types import *
import logging
from typing import cast

def test_baseline_policies(ruleset: RulesetLiteral):
    logging.critical(f'Setting up visualization folder for ruleset {ruleset}')
    vis_path = Path(f'results/baseline_policies_tests/{ruleset}')
    vis_path.mkdir(parents=True, exist_ok=True)
    def dummy_state_init(rng_key: Key) -> PolicyState:
        return jnp.zeros(1)

    env_params = EnvParams(ruleset=ruleset)
    colors = {
        'ordered': 'darkgreen',
        'closest': 'blue',
        'random': 'red',
    }

    data = {}
    for pol_name in colors.keys():
        policy = baseline_policy_register[pol_name]
        (vis_path / pol_name).mkdir(exist_ok=True)
        logging.critical(f'Starting work on policy {pol_name} for ruleset {ruleset}')
        # sanity_check(env_params, policy, dummy_state_init, 'Baseline policies', vis_path / pol_name, n_trajs=32)
        data[pol_name] = compute_running_statistics(env_params, policy, dummy_state_init, n_batches=16, batch_size=256)

    plot_cumulated_rewards(data, colors, env_params, vis_path)

if __name__ == '__main__':
    for r in ["parametric_directions", "any", "cardinal_directions"]:
        test_baseline_policies(cast(RulesetLiteral, r))
