import logging
from pathlib import Path
from src.custom_types import *
from src.baseline_policies import baseline_policy_register, make_custom_noise_level_oracle
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
from dataclasses import dataclass
from experiments.src.models.observation_policy import ObservationPolicyNetwork, ObservationPolicy

from experiments.src.eval_utils import sanity_check, compute_cumulated_rewards

experiment_output_folder = 'results/05_network_policy_analysis'

def do_sanity_checks(cfg):
    env_params = EnvParams(num_target_strokes=cfg.n_lines, max_num_strokes=2*cfg.n_lines+2, size=cfg.env_size, softness=cfg.env_softness, thickness=cfg.env_thickness)
    agent_policy_net = ObservationPolicyNetwork(decoder_config_name=cfg.decoder_config, cnn_config_name=cfg.cnn_config, seed=cfg.model_seed)

    agent_policy = ObservationPolicy(agent_policy_net)

    def dummy_state_init(rng_key: Key) -> PolicyState:
        return jnp.zeros(1)

    if cfg.noise_level > 0.:
        group_name= f'{cfg.n_lines}_lines_res_{cfg.env_size}_noise_{cfg.noise_level:.3f}_{cfg.cnn_config}_{cfg.decoder_config}_{cfg.lr:.2e}_{cfg.wd:.2e}' 
    elif cfg.env_softness == 0.03:
        group_name= f'{cfg.n_lines}_lines_res_{cfg.env_size}_{cfg.cnn_config}_{cfg.decoder_config}_{cfg.lr:.2e}_{cfg.wd:.2e}' 
    else:
        group_name= f"{cfg.env_softness}_{cfg.n_lines}_lines_res_{cfg.env_size}_noise_{cfg.noise_level:.3f}_{cfg.cnn_config}_{cfg.decoder_config}_{cfg.lr:.2e}_{cfg.wd:.2e}"


    run_name = f"{group_name}-seed_{cfg.model_seed}"
    model_path = Path("results/04_action_from_obs") / run_name / f'model_{cfg.epoch_to_load}.eqx'
    agent_policy = eqx.tree_deserialise_leaves(model_path, agent_policy)
    agent_policy = eqx.nn.inference_mode(agent_policy, value=True)
    savepath = Path(experiment_output_folder) / run_name 
    savepath.mkdir(exist_ok=True, parents=True)
    sanity_check(env_params, agent_policy, dummy_state_init, 'Agent network policy', savepath,)

    colors = {
            'network': 'blue',
            'oracle': 'darkgreen',
            'noisy_oracle_1_128': 'green',
            'noisy_oracle_003': 'teal',
            'noisy_oracle_006': 'red',
            }
    
    data = {}
    for pol_name in colors.keys():
        policy = baseline_policy_register[pol_name] if pol_name != 'network' else agent_policy
        logging.critical(f"Starting test rollouts for agent_policy {pol_name}")
        data[pol_name] = compute_cumulated_rewards(env_params, policy, dummy_state_init, n_batches=1024, batch_size=8)
    
    fig, ax = plt.subplots()
    fig.suptitle('Cumulative reward across time for baseline policies')
    fig.subplots_adjust(right=0.6)
    ax.set_xlabel('Time in trial')
    ax.set_ylabel('Cumulated reward')
    for i in range(1,cfg.n_lines+1):
        plt.axhline(i, ls=':', c="gray", alpha=.3)
    
    for pol_name, policy_arrays in data.items():
        mean = policy_arrays['mean'] 
        y_min = policy_arrays['min'] 
        y_max = policy_arrays['max'] 
        std = policy_arrays['std'] 
        # Don't show std going lower than the min (useful if only upwards variance like in oracle)
        y_low = np.max(np.stack([y_min, mean-std], -1), -1)
        y_high = np.min(np.stack([y_max, mean+std], -1), -1)
        ls = '-' if pol_name == 'network' else ':'
        alpha = 1 if pol_name == 'network' else .6
        ax.fill_between(range(2*cfg.n_lines+2), y_low, y_high, alpha=.3*alpha, ls=ls, color=colors[pol_name])
        ax.plot(mean, label=pol_name, color=colors[pol_name], alpha=alpha, ls=ls)
    ax.legend(loc='upper left', facecolor=None, edgecolor='gray', bbox_to_anchor=(1.1, 0.8),)
    fig.savefig(savepath / 'cumulated_rewards.png', dpi=600)



if __name__ == '__main__':
    @dataclass
    class RunConfig:
        model_seed: int = 0
        n_epochs: int = 50
        batches_per_epoch: int = 1000
        n_batches_val: int = 200
        lr: float = 1e-4
        wd: float = 1e-4
        n_lines: int = 4
        env_softness: float = 0.03
        # env_size: int = 128
        # batch_size: int = 16
        # cnn_config: str = "medium_deep"
        # decoder_config: str = 'small'
        # epoch_to_load: int = 50000
        # noise_level = 1/128

        env_size: int = 256
        cnn_config: str = "small"
        decoder_config: str = 'small'
        epoch_to_load: int = 42000
        noise_level = 1/256
        env_softness: float = 2/256
        env_thickness: float = 1/256

    run_config = RunConfig()
    print(f"Start run with config: \n {run_config}")
    do_sanity_checks(run_config)