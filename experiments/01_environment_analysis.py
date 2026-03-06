import time
from PIL import Image
import logging
from pathlib import Path
from src.custom_types import *
from src.single_rule_single_trial_env import online_rollout_trial_batch, offline_regenerate_observations_history
from src.basic_policies import random_agent_policy, oracle_policy, noisy_oracle_policy
import jax.numpy as jnp
import jax
import numpy as np


def sanity_check():
    '''
    Visual inspection of the trajectories (in tests/env/visualizations)
    Assertion that execution is completely deterministic once keys are fixed
    Assertion that offline regeneration of observation yields same observations as before
    Could also do speed tests here, but since policies will be traced it's not really fair
    '''

    n_reps = 2
    n_batches = 10
    save_path = Path('tests/env/visualizations')
    save_path.mkdir(exist_ok=True, parents=True)
    logging.critical("Starting test rollouts")

    def dummy_state_init(rng_key: Key) -> PolicyState:
        return jnp.zeros(1)

    B = 32
    env_params = EnvParams()

    agent_policy = random_agent_policy
    agent_state_init_fn = dummy_state_init
    ref_policy = noisy_oracle_policy
    ref_state_init_fn = dummy_state_init

    # NOTE: this version cannot yet accomodate traced policies, this will come when combining nets and env later on
    online_rollout_fn = jax.jit(
        online_rollout_trial_batch, 
        static_argnums=(2, 3, 4, 5, 6, 7, 8)
    )

    offline_regenerate_fn = jax.jit(
        offline_regenerate_observations_history, 
        static_argnums=(1,)
    )


    times = []
    for batch in range(n_batches):
        env_key = jax.random.key(777+(batch%2))
        pol_key = jax.random.key(777+batch)
        batch_history: HistoryOutput = online_rollout_fn(# Traced
                                        env_key, pol_key, 
                                        # Static
                                        agent_policy, agent_state_init_fn, 
                                        ref_policy, ref_state_init_fn,
                                        B, env_params)
        
        ref_state_history: HistoryEnvState = HistoryOutput.env_state
        tic = time.time()
        times.append(time.time()-tic)
        ref_obs: HistoryCanvas = batch_history.obs
        ref_obs: Float[Array, "B T H W 3"] = ref_obs.transpose(1,0,3,4,2)
        pixels = (256* np.asarray(ref_obs, dtype=np.float64)).astype(np.uint8)
        logging.critical(f"Batch {batch} done in {times[-1]:.5f}s, total rewards: {np.asarray(batch_history.env_state.step_reward, dtype=np.float64).sum()}")

        for b_idx in range(5):
            # TODO: add visualization for the reward
            images = [Image.fromarray(pixels[b_idx,t]) for t in range(pixels.shape[1])]
            images[0].save(save_path / f'sanity_check_batch_{batch}_trajectory_{b_idx}.gif', save_all=True,
            append_images=images[1:], # append rest of the images
            duration=1000, # in milliseconds
            loop=0)

        for rep in range(n_reps):
            # Should be same env every two reps, but different realizations of the policy
            repeat_online = online_rollout_fn(env_key, pol_key, agent_policy, agent_state_init_fn, ref_policy, ref_state_init_fn, B, env_params)
            compare_online = repeat_online.obs.transpose(1,0,3,4,2)
            assert np.allclose(ref_obs, compare_online), "Repeating the same rollout yields different results !"
            repeat_offline_obs = offline_regenerate_fn(ref_state_history, env_params)
            compare_offline = repeat_offline_obs.transpose(1,0,3,4,2)
            assert np.allclose(ref_obs, compare_offline), "Regenerating observations from EnvState history yields different results !"


if __name__ == '__main__':
    sanity_check()