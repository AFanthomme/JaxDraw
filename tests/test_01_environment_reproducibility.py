import jax
# Forces strict fp32 math across all GPU architectures
jax.config.update("jax_default_matmul_precision", "float32")
import chex 
import pytest
from experiments.src.custom_types import *
from experiments.src.single_rule_single_trial_env import on_policy_online_rollout, off_policy_online_rollout, offline_regenerate_observations_history, offline_replay_actions
from experiments.src.baseline_policies import random_agent_policy, noisy_oracle_policy

B = 16

@pytest.fixture(scope="module")
def common_setup():
    env_params = EnvParams()
    dummy_state_init = lambda x: jax.numpy.zeros(1)
    online_rollout_fn = jax.jit(off_policy_online_rollout, static_argnums=(2, 3, 4, 5, 6, 7))
    onpolicy_rollout_fn = jax.jit(on_policy_online_rollout, static_argnums=(2, 3, 4, 5,))
    offline_replay_actions_fn = jax.jit(offline_replay_actions, static_argnums=(2,))
    offline_regenerate_obs_fn = jax.jit(offline_regenerate_observations_history, static_argnums=(1,))
    return env_params, dummy_state_init, online_rollout_fn, onpolicy_rollout_fn, offline_replay_actions_fn, offline_regenerate_obs_fn

@pytest.fixture(params=range(10))
def reference_rollout_data(request, common_setup):
    """Generates the reference rollout for a specific seed."""
    batch_idx = request.param
    env_params, dummy_state_init, online_rollout_fn, _, _, _ = common_setup
    
    env_key = jax.random.key(777 + batch_idx)
    pol_key = jax.random.key(777 + batch_idx)

    # Generate the ground-truth rollout
    # Putting different policies in teacher/agent covered in experimetns, here use same to compare to onpolicy
    ref_rollout = online_rollout_fn(
        env_key, pol_key, 
        noisy_oracle_policy, dummy_state_init, 
        noisy_oracle_policy, dummy_state_init,
        B, env_params
    )
    
    # Return a dictionary containing everything downstream tests might need
    return {
        "env_key": env_key,
        "pol_key": pol_key,
        "ref_rollout": ref_rollout
    }

def test_rerun_online_determinism(common_setup, reference_rollout_data):
    """Ensures that calling the JAX function again yields identical results."""
    env_params, dummy_state_init, online_rollout_fn, _,  _, _ = common_setup
    data = reference_rollout_data
    
    repeat_online = online_rollout_fn(
        data["env_key"], data["pol_key"], 
        noisy_oracle_policy, dummy_state_init, 
        noisy_oracle_policy, dummy_state_init, 
        B, env_params
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
    # Replayed using teacher, so compare to teacher rewards
    ref_rewards: RewardHistory = reference_rollout_data["ref_rollout"].teacher_reward
    chex.assert_trees_all_close(replayed_rewards_history, ref_rewards)

def test_offline_observation_regeneration(common_setup, reference_rollout_data, do_offline_replay):
    env_params, _, _, _, _, offline_regenerate_obs_fn = common_setup
    ref_obs = reference_rollout_data["ref_rollout"].obs
    replayed_state_history, _ = do_offline_replay
    replayed_obs = offline_regenerate_obs_fn(replayed_state_history, env_params)
    # We don't need that much precision on the observations themselves, this will fail at default 1e-6 if enabling tf32
    chex.assert_trees_all_close(ref_obs, replayed_obs, rtol=1e-5, atol=1e-5)

def test_onpolicy_substitution(common_setup, reference_rollout_data):
    env_params, dummy_state_init, _, on_policy_online_rollout, _, _ = common_setup
    data = reference_rollout_data
    
    repeat_onpolicy = on_policy_online_rollout(
        data["env_key"], data["pol_key"], 
        noisy_oracle_policy, dummy_state_init, 
        B, env_params
    )
    
    chex.assert_trees_all_close(data["ref_rollout"], repeat_onpolicy)
