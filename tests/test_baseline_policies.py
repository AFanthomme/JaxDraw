from pathlib import Path
from src.config import EnvParams, RULESET_OPTIONS, RulesetLiteral
from src.baseline_policies import baseline_policy_register
import jax.numpy as jnp
from src.eval_utils import compute_running_statistics
from src.image_utils import plot_cumulated_rewards
from src.custom_types import RulesetLiteral, PolicyState, Key
import pytest

@pytest.mark.parametrize("ruleset", RULESET_OPTIONS)
def test_baseline_policies(ruleset: RulesetLiteral):
    vis_path = Path(f'results/baseline_policies_tests/{ruleset}')
    vis_path.mkdir(parents=True, exist_ok=True)
    def dummy_state_init(rng_key: Key) -> PolicyState:
        return jnp.zeros(1)

    env_params = EnvParams(ruleset=ruleset)
    colors = {
        'ordered': 'darkgreen',
        'closest': 'orange',
        'random': 'red',
    }

    data = {}
    for pol_name in colors.keys():
        policy = baseline_policy_register[pol_name]
        (vis_path / pol_name).mkdir(exist_ok=True)
        data[pol_name] = compute_running_statistics(env_params, policy, dummy_state_init, n_batches=64, batch_size=256)

    plot_cumulated_rewards(data, colors, env_params, vis_path)

    assert True
