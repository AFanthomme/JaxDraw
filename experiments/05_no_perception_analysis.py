import logging
from pathlib import Path
from src.custom_types import *
from src.baseline_policies import baseline_policy_register, make_custom_noise_level_policy
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
from dataclasses import dataclass
from experiments.src.models.envstate_policy import build_envstate_policy, EnvStatePolicy

from experiments.src.eval_utils import sanity_check, compute_cumulated_rewards
from experiments.src.image_utils import plot_cumulated_rewards



def do_sanity_checks(cfg):
    experiment_output_folder = f"results/no_perception/{cfg.ruleset}"
    env_params = EnvParams(ruleset=cfg.ruleset, num_target_strokes=cfg.n_lines, max_num_strokes=2*cfg.n_lines+2, size=cfg.env_size, softness=cfg.env_softness, thickness=cfg.env_thickness)

    def dummy_state_init(rng_key: Key) -> PolicyState:
        return jnp.zeros(1)
    group_name= f"{cfg.model_config}_noise_{cfg.noise_level:.3f}_B_{cfg.batch_size}_{cfg.lr:.2e}_{cfg.wd:.2e}" 
    run_name = f"{group_name}-seed_{cfg.model_seed}"
    model = build_envstate_policy(config_name=cfg.model_config, seed=cfg.model_seed)
    model_path = Path(experiment_output_folder) / run_name / f'model_{cfg.epoch_to_load}.eqx'
    agent_policy = eqx.tree_deserialise_leaves(model_path, model)
    agent_policy = eqx.nn.inference_mode(agent_policy, value=True)
    agent_policy = EnvStatePolicy(agent_policy)
    savepath = Path(experiment_output_folder) / run_name / "visualizations"
    savepath.mkdir(exist_ok=True, parents=True)
    sanity_check(env_params, agent_policy, dummy_state_init, 'Agent network policy (no perception)', savepath,)

    colors = {
            'network': 'blue',
            'noisy_closest': 'orange',
            'noisy_ordered': 'darkgreen',
            }
    
    data = {}
    for pol_name in colors.keys():
        policy = baseline_policy_register[pol_name] if pol_name != 'network' else agent_policy
        logging.critical(f"Starting test rollouts for agent_policy {pol_name}")
        data[pol_name] = compute_cumulated_rewards(env_params, policy, dummy_state_init, n_batches=1024, batch_size=8)
    
    plot_cumulated_rewards(data, colors, env_params, savepath)



if __name__ == '__main__':
    @dataclass
    class RunConfig:
        model_seed: int = 0
        adam_eps: float = 1e-6
        grad_clip: float = 1.
        n_lines: int = 4

        wd: float = 0.
        env_size: int = 128
        lr: float = 3e-4
        batch_size: int = 256
        model_config: str = 'medium'
        # noise_level = 1/128
        noise_level = 0.
        env_softness: float = 2/128
        env_thickness: float = 1/128
        n_epochs: int = 100
        warmup_steps = 100
        n_batches_val: int = 200 
        batches_per_epoch: int = 500
        ruleset = 'along_parametric_directions_with_decreasing'
        # ruleset = 'along_cardinal_directions'
        epoch_to_load = 100_000

    run_config = RunConfig()
    print(f"Start run with config: \n {run_config}")
    do_sanity_checks(run_config)