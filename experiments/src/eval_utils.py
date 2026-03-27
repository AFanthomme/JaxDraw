from src.custom_types import *
import jax
import numpy as np
import jax.numpy as jnp
from experiments.src.image_utils import save_gif
from pathlib import Path
from src.single_rule_single_trial_env import on_policy_online_rollout, off_policy_online_rollout
from experiments.src.stat_utils import RunningStats
import equinox as eqx

def sanity_check(env_params: EnvParams, agent_policy: Policy, agent_state_init: PolicyStateInitializer, 
                 title: str, savepath: Path, n_trajs: int=8, seed: int=777, static_agent: Bool=True, static_teacher: Bool=True,
                 teacher_policy: Optional[Policy]=None, teacher_state_init: Optional[PolicyStateInitializer]=None):

    assert savepath.exists(), "Please create the output directory for sanity_check before calling it !"
    
    env_key = jax.random.key(seed)
    pol_key = jax.random.key(seed)

    if teacher_policy is None:
        online_rollout_fn = eqx.filter_jit(on_policy_online_rollout)
        full_rollout = online_rollout_fn(# Traced
                                    env_key, pol_key, 
                                    # Static
                                    agent_policy, agent_state_init, 
                                    n_trajs, env_params)
    else:
        assert teacher_state_init is not None, "Provide teacher init fn if you provide teacher_policy!"
        offline_rollout_fn = eqx.filter_jit(off_policy_online_rollout)
        full_rollout = offline_rollout_fn(# Traced
                                    env_key, pol_key, 
                                    # Static
                                    agent_policy, agent_state_init, 
                                    teacher_policy, teacher_state_init, 
                                    n_trajs, env_params)
    
    state_history: EnvStateHistory = full_rollout.env_state
    # Put the Batch dimension first since we don't plot in parallel
    agent_actions : Float["Array", "B T 3"] = full_rollout.agent_action.transpose(1, 0, 2)
    teacher_actions : Float["Array", "B T 3"] = full_rollout.teacher_action.transpose(1, 0, 2)
    positions: Float["Array", "B T 2"] = state_history.position.transpose(1, 0, 2)
    agent_rewards: Float["Array", "B T"] = full_rollout.agent_reward.transpose(1, 0)
    teacher_rewards: Float["Array", "B T"] = full_rollout.teacher_reward.transpose(1, 0)
    # For obs, need to put the RGB at the end and switch x,y for correct display
    obs: Float["Array", "B T H W 3"] = full_rollout.obs.transpose(1, 0, 4, 3, 2)

    for b in range(n_trajs):
        if teacher_policy is not None:
            save_gif(savepath / f'traj_{b}.gif', obs[b], positions[b], agent_actions[b], agent_rewards[b], teacher_actions[b], teacher_rewards[b], title=title)
        else:
            save_gif(savepath / f'traj_{b}.gif', obs[b], positions[b], agent_actions[b], agent_rewards[b], None, None, title=title)

def compute_cumulated_rewards(env_params: EnvParams, policy: Policy, state_init: PolicyStateInitializer, n_batches: int=128, batch_size: int=128, seed: int=777):
    T = env_params.max_num_strokes
    online_rollout_fn = eqx.filter_jit(on_policy_online_rollout)
    running_stats = RunningStats(shape=(T,))

    for b in range(n_batches):
        env_key = jax.random.key(seed+b)
        pol_key = jax.random.key(seed+b)
        batch_rollout = online_rollout_fn(# Traced
                                env_key, pol_key, 
                                # Static
                                policy, state_init, 
                                batch_size, env_params)
        
        batch_rewards: Float["Array", "T B"] = batch_rollout.agent_reward
        cumulated_rewards = jnp.cumsum(batch_rewards, axis=0).T
        running_stats.update(cumulated_rewards)


    out_dict = {'mean': np.asarray(running_stats.mean), 'std': np.sqrt(np.asarray(running_stats.var)), 
                        'min': np.asarray(running_stats.min), 'max': np.asarray(running_stats.max)}
    
    return out_dict


