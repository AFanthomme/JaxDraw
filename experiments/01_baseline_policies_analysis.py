from PIL import Image
import logging
from pathlib import Path
from src.custom_types import *
from src.single_rule_single_trial_env import on_policy_online_rollout, off_policy_online_rollout
from src.baseline_policies import baseline_policy_register
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from experiments.src.utils import RunningStats

def save_gif(savepath, images, agent_actions, teacher_actions, teacher_rewards, agent_rewards, positions, title=''):  
    T, H, W, _ = images.shape 

    fig, ax = plt.subplots()
    ax.axis('off')
    fig.suptitle(title)
    fig.subplots_adjust(right=0.6)
    img_display = ax.imshow(images[0], animated=True, extent=[0, 1, 0, 1], origin='lower')
    agent_line, = ax.plot([], [], color='yellow', ls=':', label=f'Agent: p=     ', linewidth=2, animated=True)
    reward_for_legend, = ax.plot([], [], color='yellow', label=f'Agent reward:      ', linewidth=2, animated=True)
    teacher_line, = ax.plot([], [], color='pink', label=f'Teacher: p=     ', linewidth=2, animated=True)
    reward_for_legend, = ax.plot([], [], color='pink', label=f'Teacher reward:      ', linewidth=2, animated=True)
    start_dots = ax.scatter([], [], color='yellow', s=60, zorder=5, animated=True, label='Start pos')
    end_dots = ax.scatter([], [], color='gray', s=60, zorder=5, animated=True, label='Next pos')
    leg = ax.legend(loc='upper left', facecolor=None, edgecolor='gray', bbox_to_anchor=(1.1, 0.8),)

    def update_frame(t):
        img_display.set_data(images[t])
        x, y = positions[t]
        dxa, dya, pa =  agent_actions[t]
        dxt, dyt, pt = teacher_actions[t]
        nx, ny = positions[min(t+1, len(positions))]

        agent_line.set_data([x, x+dxa], [y, y+dya])
        teacher_line.set_data([x, x+dxt], [y, y+dyt])
        leg.get_texts()[0].set_text(f'agent: p={pa:.2f}')
        leg.get_texts()[1].set_text(f'agent reward: {agent_rewards[t]:.2f}')
        leg.get_texts()[2].set_text(f'teacher: p={pt:.2f}')
        leg.get_texts()[3].set_text(f'teacher reward: {teacher_rewards[t]:.2f}')
        start_dots.set_offsets([x, y])
        end_dots.set_offsets([nx, ny])

        return img_display, agent_line, teacher_line, reward_for_legend, start_dots, end_dots, leg 
    
    anim = FuncAnimation(fig, update_frame, frames=T, blit=True)
    anim.save(savepath, writer=PillowWriter(fps=1))
    plt.close()


def sanity_check(agent_policy_name: str="noisy_oracle_policy", teacher_policy_name: Optional[str]=None):
    '''
    Visual inspection of the trajectories; two realizations of the environment, first one with 2 different realizations of the (noisy) policy.
    '''
    B = 4
    logging.critical(f"Starting test rollouts for agent_policy {agent_policy_name}, teacher {teacher_policy_name}")

    def dummy_state_init(rng_key: Key) -> PolicyState:
        return jnp.zeros(1)

    env_params = EnvParams()
    agent_policy = baseline_policy_register[agent_policy_name]

    if teacher_policy_name is None:
        online_rollout_fn = jax.jit(on_policy_online_rollout, static_argnums=(2, 3, 4, 5))
        title = f'On-policy, name: {agent_policy_name}'
        base_savepath = Path(f'results/01_baseline_policies/onpolicy_{agent_policy_name}')
    else:
        offline_rollout_fn = jax.jit(off_policy_online_rollout, static_argnums=(2, 3, 4, 5, 6, 7))
        teacher_policy = baseline_policy_register[teacher_policy_name]
        title = f'Off-policy, agent: {agent_policy_name}, teacher {teacher_policy_name}'
        base_savepath = Path(f'results/01_baseline_policies/offpolicy_agent_{agent_policy_name}_teacher_{teacher_policy_name}')

    base_savepath.mkdir(exist_ok=True, parents=True)

    env_key = jax.random.key(777)
    pol_key = jax.random.key(777)

    if teacher_policy_name is None:
        full_rollout = online_rollout_fn(# Traced
                                    env_key, pol_key, 
                                    # Static
                                    agent_policy, dummy_state_init, 
                                    B, env_params)
    else:
        full_rollout = offline_rollout_fn(# Traced
                                    env_key, pol_key, 
                                    # Static
                                    agent_policy, dummy_state_init, 
                                    teacher_policy, dummy_state_init, 
                                    B, env_params)
    

    state_history: EnvStateHistory = full_rollout.env_state
    # Put the Batch dimension first since we don't plot in parallel
    agent_actions : Float["Array", "B T 3"] = full_rollout.agent_action.transpose(1, 0, 2)
    teacher_actions : Float["Array", "B T 3"] = full_rollout.teacher_action.transpose(1, 0, 2)
    positions: Float["Array", "B T 2"] = state_history.position.transpose(1, 0, 2)
    agent_rewards: Float["Array", "B T"] = full_rollout.agent_reward.transpose(1, 0)
    teacher_rewards: Float["Array", "B T"] = full_rollout.teacher_reward.transpose(1, 0)
    # For obs, need to put the RGB at the end and switch x,y for correct display
    obs: Float["Array", "B T H W 3"] = full_rollout.obs.transpose(1, 0, 4, 3, 2)

    for b in range(B):
        savepath = base_savepath / f'traj_{b}.gif'
        save_gif(savepath, obs[b], agent_actions[b], teacher_actions[b], agent_rewards[b], teacher_rewards[b], positions[b], title=title)



def cumulated_rewards_baselines():
    B = 128
    # n_batches = 1024
    n_batches = 128

    env_params = EnvParams()
    T=env_params.max_num_strokes
    R_max = env_params.num_target_strokes
    def dummy_state_init(rng_key: Key) -> PolicyState:
        return jnp.zeros(1)
    
    colors = {
            'random': 'r',
            'oracle': 'darkgreen',
            'noisy_oracle_003': 'olive',
            'noisy_oracle_004': 'teal',
            'noisy_oracle_006': 'orange'
            }
    
    data = {}
    for pol_name in colors.keys():
        policy = baseline_policy_register[pol_name]
        logging.critical(f"Starting test rollouts for agent_policy {pol_name}")
        running_stats = RunningStats(shape=(T))

        # Need to recompile after every policy change if we want both speedup and actual change

        online_rollout_fn = jax.jit(on_policy_online_rollout, static_argnums=(2, 3, 4, 5))
        for b in range(n_batches):
            env_key = jax.random.key(777+b)
            pol_key = jax.random.key(777+b)
            batch_rollout = online_rollout_fn(# Traced
                                    env_key, pol_key, 
                                    # Static
                                    policy, dummy_state_init, 
                                    B, env_params)
            
            batch_rewards: Float["Array", "T B"] = batch_rollout.agent_reward

            # Quick sanity check that we don't get panalty at the same time as reward
            pos_reward = np.asarray(batch_rewards) > 0.
            integer_reward = np.isin(np.asarray(batch_rewards),  [0, 1, 2, 3, 4])

            if np.any(pos_reward * ~integer_reward):
                logging.critical(f'Found rewards that is neither integer nor negative: {batch_rewards[pos_reward * ~integer_reward]}')

            cumulated_rewards = jnp.cumsum(batch_rewards, axis=0).T
            running_stats.update(np.asarray(cumulated_rewards))


        data[pol_name] = {'mean': np.asarray(running_stats.mean), 'std': np.sqrt(np.asarray(running_stats.var)), 
                          'min': np.asarray(running_stats.min), 'max': np.asarray(running_stats.max)}

    fig, ax = plt.subplots()
    fig.suptitle('Cumulative reward across time for baseline policies')
    fig.subplots_adjust(right=0.6)
    ax.set_xlabel('Time in trial')
    ax.set_ylabel('Cumulated reward')
    for i in range(1,5):
        plt.axhline(i, ls=':', c="gray", alpha=.3)
    

    for pol_name, policy_arrays in data.items():
        mean = policy_arrays['mean'] 
        y_min = policy_arrays['min'] 
        y_max = policy_arrays['max'] 
        std = policy_arrays['std'] 
        # Don't show std going lower than the min (useful if only upwards variance like in oracle)
        y_low = np.max(np.stack([y_min, mean-std], -1), -1)
        y_high = np.min(np.stack([y_max, mean+std], -1), -1)
        ax.fill_between(range(T), y_low, y_high, alpha=.3, color=colors[pol_name])
        ax.plot(mean, label=pol_name, color=colors[pol_name])
    # ax.set_ylim(0, R_max+.5)
    ax.legend(loc='upper left', facecolor=None, edgecolor='gray', bbox_to_anchor=(1.1, 0.8),)
    fig.savefig('results/01_baseline_policies/cumulated_rewards.png', dpi=600)





if __name__ == '__main__':
    cumulated_rewards_baselines()
    for k in baseline_policy_register.keys():
        sanity_check(agent_policy_name=k)
    sanity_check(agent_policy_name='oracle', teacher_policy_name='noisy_oracle')
    sanity_check(agent_policy_name='noisy_oracle', teacher_policy_name='random')
    sanity_check(agent_policy_name='random', teacher_policy_name='oracle')
