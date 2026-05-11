import jax
jax.config.update("jax_default_matmul_precision", "float32")
import chex 
import pytest
from src.custom_types import *
from src.environment import on_policy_online_rollout, off_policy_online_rollout, offline_regenerate_observations_history, offline_replay_actions
from src.baseline_policies import baseline_policy_register, make_custom_noise_level_policy

B = 16
@pytest.fixture(scope="module")
def common_setup():
    env_params = EnvParams()
    dummy_state_init = lambda x: jax.numpy.zeros(1)
    offpolicy_rollout_fn = jax.jit(off_policy_online_rollout, static_argnums=(2, 3, 4, 5, 6, 7, 8))
    onpolicy_rollout_fn = jax.jit(on_policy_online_rollout, static_argnums=(2, 3, 4, 5, 6))
    offline_replay_actions_fn = jax.jit(offline_replay_actions, static_argnums=(2,))
    offline_regenerate_obs_fn = jax.jit(offline_regenerate_observations_history, static_argnums=(1,))
    return env_params, dummy_state_init, offpolicy_rollout_fn, onpolicy_rollout_fn, offline_replay_actions_fn, offline_regenerate_obs_fn

@pytest.fixture(params=range(10))
def reference_rollout_data(request, common_setup):
    """Generates the reference rollout for a specific seed."""
    batch_idx = request.param
    env_params, dummy_state_init, offpolicy_rollout_fn, _, _, _ = common_setup
    
    env_key = jax.random.key(777 + batch_idx)
    pol_key = jax.random.key(777 + batch_idx)

    on_policy = make_custom_noise_level_policy(baseline_policy_register['ordered'], 0.01)
    off_policy = baseline_policy_register['random']

    ref_rollout = offpolicy_rollout_fn(
        env_key, pol_key, 
        on_policy, dummy_state_init, 
        off_policy, dummy_state_init, 
        B, env_params
    )

    return {
        "on_policy": on_policy,
        "off_policy": off_policy,
        "env_key": env_key,
        "pol_key": pol_key,
        "ref_rollout": ref_rollout
    }

@pytest.fixture(params=(True, False))
def test_rollout_determinism(request, common_setup, reference_rollout_data):
    is_visual_env = request.param
    env_params, dummy_state_init, offpolicy_rollout_fn, _, _, _ = common_setup
    data = reference_rollout_data
    repeat_online = offpolicy_rollout_fn(
        data["env_key"], data["pol_key"], 
        data["on_policy"], dummy_state_init, 
        data["off_policy"], dummy_state_init, 
        B, env_params, is_visual_env
    )
    
    chex.assert_trees_all_close(data["ref_rollout"], repeat_online)
    
@pytest.fixture
def do_offline_replay(common_setup, reference_rollout_data):
    env_params, _, _, _, offline_replay_actions_fn, _  = common_setup
    env_key = reference_rollout_data["env_key"]
    ref_rollout = reference_rollout_data["ref_rollout"]
    ref_actions = ref_rollout.teacher_action
    replayed_state_history, replayed_rewards = offline_replay_actions_fn(env_key, ref_actions, env_params)
    return replayed_state_history, replayed_rewards

def test_offline_action_replay_states_history(reference_rollout_data, do_offline_replay):
    replayed_state_history, _ = do_offline_replay
    ref_state_history = reference_rollout_data["ref_rollout"].env_state
    chex.assert_trees_all_close(replayed_state_history, ref_state_history)

def test_offline_action_replay_rewards_history(reference_rollout_data, do_offline_replay):
    _, replayed_rewards_history = do_offline_replay
    ref_rewards: RewardHistory = reference_rollout_data["ref_rollout"].teacher_reward
    chex.assert_trees_all_close(replayed_rewards_history, ref_rewards)

def test_offline_observation_regeneration(common_setup, reference_rollout_data, do_offline_replay):
    env_params, _, _, _, _, offline_regenerate_obs_fn = common_setup
    ref_obs = reference_rollout_data["ref_rollout"].obs
    replayed_state_history, _ = do_offline_replay
    replayed_obs = offline_regenerate_obs_fn(replayed_state_history, env_params)
    chex.assert_trees_all_close(ref_obs, replayed_obs, rtol=1e-5, atol=1e-5)
