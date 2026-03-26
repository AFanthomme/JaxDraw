import logging
from src.custom_types import *
from src.baseline_policies import baseline_policy_register
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from experiments.src.eval_utils import sanity_check, compute_cumulated_rewards


def do_sanity_check(agent_policy_name: str, teacher_policy_name: Optional[str]=None):
    env_params = EnvParams()
    def dummy_state_init(rng_key: Key) -> PolicyState:
        return jnp.zeros(1)

    agent_policy = baseline_policy_register[agent_policy_name]

    if teacher_policy_name is None:
        teacher_policy = None
        title = f'On-policy, name: {agent_policy_name}'
        savepath = Path(f'results/01_baseline_policies/onpolicy_{agent_policy_name}')
    else:
        teacher_policy = baseline_policy_register[teacher_policy_name]
        title = f'Off-policy, agent: {agent_policy_name}, teacher {teacher_policy_name}'
        savepath = Path(f'results/01_baseline_policies/offpolicy_agent_{agent_policy_name}_teacher_{teacher_policy_name}')

    savepath.mkdir(exist_ok=True, parents=True)
    sanity_check(env_params, agent_policy, dummy_state_init, title, savepath, teacher_policy=teacher_policy, teacher_state_init=dummy_state_init)

def cumulated_rewards_baselines():
    env_params = EnvParams()
    T = env_params.max_num_strokes
    def dummy_state_init(rng_key: Key) -> PolicyState:
        return jnp.zeros(1)
    
    colors = {
            'oracle': 'darkgreen',
            'noisy_oracle_001': 'olive',
            'noisy_oracle_003': 'teal',
            'noisy_oracle_006': 'orange',
            'random': 'r',
            }
    
    data = {}
    for pol_name in colors.keys():
        policy = baseline_policy_register[pol_name]
        logging.critical(f"Starting test rollouts for agent_policy {pol_name}")
        data[pol_name] = compute_cumulated_rewards(env_params, policy, dummy_state_init)

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
    ax.legend(loc='upper left', facecolor=None, edgecolor='gray', bbox_to_anchor=(1.1, 0.8),)
    fig.savefig('results/01_baseline_policies/cumulated_rewards.png', dpi=600)

if __name__ == '__main__':
    Path('results/01_baseline_policies/').mkdir(exist_ok=True)
    # for k in baseline_policy_register.keys():
    #     do_sanity_check(agent_policy_name=k)
    # do_sanity_check(agent_policy_name='oracle', teacher_policy_name='noisy_oracle')
    # do_sanity_check(agent_policy_name='noisy_oracle', teacher_policy_name='random')
    # do_sanity_check(agent_policy_name='random', teacher_policy_name='oracle')
    cumulated_rewards_baselines()
