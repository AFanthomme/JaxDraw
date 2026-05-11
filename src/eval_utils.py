import jax
import numpy as np
import jax.numpy as jnp
import equinox as eqx
from src.image_utils import save_sequence_as_gif
from pathlib import Path
from src.custom_types import *
from src.environment import on_policy_online_rollout, off_policy_online_rollout
from src.stat_utils import RunningStats
from typing import Optional, cast

def sanity_check(env_params: EnvParams, agent_policy: Policy, agent_state_init: PolicyStateInitializer, 
                 title: str, savepath: Path, n_trajs: int=32, seed: int=777, static_agent: Bool=True, static_teacher: Bool=True,
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

    cast(FullRollout, full_rollout)
    
    for b in range(n_trajs):
        env_states = jax.tree_util.tree_map(lambda x: x[:, b], full_rollout.env_state).as_type(EnvStateSequence)
        images = full_rollout.obs[:,b]
        agent_actions = full_rollout.agent_action[:,b]
        teacher_actions = full_rollout.teacher_action[:,b]
        agent_rewards = full_rollout.teacher_reward[:,b]
        teacher_rewards = full_rollout.agent_reward[:,b]
        if teacher_policy is None:
            save_sequence_as_gif(savepath / f'traj_{b}.gif', images, env_states, agent_actions, agent_rewards, None, None, title=title)
        else:
            save_sequence_as_gif(savepath / f'traj_{b}.gif', images, env_states, agent_actions, agent_rewards, teacher_actions, teacher_rewards, title=title)

def compute_running_statistics(env_params: EnvParams, policy: Policy, state_init: PolicyStateInitializer, n_batches: int=128, batch_size: int=128, seed: int=777):
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
                        'min': np.asarray(running_stats.min), 'max': np.asarray(running_stats.max),}
    
    return out_dict


