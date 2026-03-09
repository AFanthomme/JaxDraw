# import time
# from PIL import Image
# import logging
# from pathlib import Path
# from src.custom_types import *
# from src.single_rule_single_trial_env import off_policy_online_rollout, offline_regenerate_observations_history, offline_replay_actions
# from src.basic_policies import random_agent_policy, oracle_policy, noisy_oracle_policy
# import jax.numpy as jnp
# import jax
# import numpy as np

# def sanity_check():
#     '''
#     Visual inspection of the trajectories; two realizations of the environment, first one with 2 different realizations of the (noisy) policy.
#     '''
#     B = 16
#     n_batches = 3
#     images_path = Path('results/sanity_checks/env')
#     images_path.mkdir(exist_ok=True, parents=True)
#     logging.critical("Starting test rollouts")

#     def dummy_state_init(rng_key: Key) -> PolicyState:
#         return jnp.zeros(1)

#     env_params = EnvParams()

#     agent_policy = noisy_oracle_policy
#     agent_state_init_fn = dummy_state_init
#     ref_policy = noisy_oracle_policy
#     ref_state_init_fn = dummy_state_init

#     # NOTE: this version cannot yet accomodate traced policies, this will come when combining nets and env later on
#     online_rollout_fn = jax.jit(
#         off_policy_online_rollout, 
#         static_argnums=(2, 3, 4, 5, 6, 7)
#     )

#     times = []
#     for batch in range(n_batches):
#         # Show two realizations of the
#         env_key = jax.random.key(777+(batch%2))
#         pol_key = jax.random.key(777+(batch//2))
#         tic = time.time()
#         full_rollout = online_rollout_fn(# Traced
#                                         env_key, pol_key, 
#                                         # Static
#                                         agent_policy, agent_state_init_fn, 
#                                         ref_policy, ref_state_init_fn,
#                                         B, env_params)
#         times.append(time.time()-tic)
        
#         ref_actions : ActionHistory = full_rollout.teacher_action
#         ref_state_history: EnvStateHistory = full_rollout.env_state
#         ref_rewards: RewardHistory = full_rollout.reward

#         ref_obs: CanvasHistory = full_rollout.obs
#         ref_obs: Float[Array, "B T H W 3"] = ref_obs.transpose(1,0,3,4,2)
#         pixels = (256* np.asarray(ref_obs, dtype=np.float64)).astype(np.uint8)
#         logging.critical(f"Batch {batch} done in {times[-1]:.5f}s, total rewards: {np.asarray(ref_rewards, dtype=np.float64).sum()}")

#         for b_idx in range(min(B,5)):
#             # TODO: add visualization for the reward
#             images = [Image.fromarray(pixels[b_idx,t]) for t in range(pixels.shape[1])]
#             images[0].save(images_path / f'env_{(batch%2)}_pol_{batch//2}_trajectory_{b_idx}.gif', save_all=True,
#             append_images=images[1:], # append rest of the images
#             duration=1000, # in milliseconds
#             loop=0)

# def analyze_noise_sensitivity():
#     '''
#     Replay the same trajectories, but with some noise applied to oracle movement to confirm sensitivity

#     For (gaussian) noise level n, error after t steps typical error grows like n*sqrt(t) (max more like 3*n*sqrt(t))
#     When this crosses quality_floor, start to get rewards misses

#     Therefore, plot (with mean/std across many rollouts) cumulative_reward_up_to_t for different levels of noise
#     Use that to confirm lowest acceptable error level for cloning  
#     '''
#     B = 32
#     n_reps = 10
#     n_batches = 1000 
#     env_params = EnvParams()
#     T = env_params.max_num_strokes
#     noise_levels = [1e-6, 1e-4, 1e-3, 1e-2]
#     noisy_reward_trajectories = jnp.zeros((len(noise_levels), n_batches, n_reps, B, T))
#     ref_reward_trajectories = jnp.zeros((n_batches, B, T))

#     save_path = Path('tests/env/noise_sensitivity')
#     save_path.mkdir(exist_ok=True, parents=True)
#     logging.critical("Starting test rollouts")

#     def dummy_state_init(rng_key: Key) -> PolicyState:
#         return jnp.zeros(1)


#     agent_policy = random_agent_policy
#     agent_state_init_fn = dummy_state_init
#     ref_policy = oracle_policy
#     ref_state_init_fn = dummy_state_init

#     # NOTE: this version cannot yet accomodate traced policies, this will come when combining nets and env later on
#     online_rollout_fn = jax.jit(
#         off_policy_online_rollout, 
#         static_argnums=(2, 3, 4, 5, 6, 7, 8)
#     )

#     offline_regenerate_fn = jax.jit(
#         offline_regenerate_observations_history, 
#         static_argnums=(1,)
#     )


#     times = []
#     for batch in range(n_batches):
#         env_key = jax.random.key(777+batch)
#         pol_key = jax.random.key(777+batch)
#         full_rollout: FullRollout = online_rollout_fn(# Traced
#                                         env_key, pol_key, 
#                                         # Static
#                                         agent_policy, agent_state_init_fn, 
#                                         ref_policy, ref_state_init_fn,
#                                         B, env_params)
        
#         ref_state_history: EnvStateHistory = full_rollout.env_state
#         ref_reward_trajectories[batch] = ref_state_history.reward.transpose(0,1)
#         ref_actions: ActionHistory = full_rollout.teacher_action

#         for rep in range(n_reps):
#             rep_key = jax.random.key(7777 + rep)
#             for noise_idx, noise_level in enumerate(noise_levels):
#                 noise_array = jax.random.uniform(rep_key, ref_actions.shape, minval=-1, maxval=1)
#                 # Don't perturb the pressure part of the action
#                 perturbed_actions = ref_actions + noise_array * jnp.array([.9, .9, 0.])[None, None, :]
#                 perturbed_env_history: EnvStateHistory = offline_replay_actions(env_key, perturbed_actions, env_params)
#                 noisy_reward_trajectories[noise_idx, batch, rep] = perturbed_env_history.reward.transpose(0,1)

#     # Cumulative rewards :
#     ref_reward_trajectories = np.asarray(ref_reward_trajectories)
#     ref_reward_trajectories = np.cumsum(ref_reward_trajectories.reshape((-1, T)), -1)

#     noisy_reward_trajectories = np.asarray(noisy_reward_trajectories)
#     noisy_reward_trajectories = np.cumsum(noisy_reward_trajectories.reshape((len(noise_levels), -1, T)), -1)

#     # Do the actual plots

# if __name__ == '__main__':
#     sanity_check()