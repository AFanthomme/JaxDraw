import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from custom_types import *
import time
import logging
from PIL import Image
from pathlib import Path
import numpy as np
from single_rule_single_trial_env import oracle_policy, rollout_trial_batch
from basic_net import BasicCNN
from functools import partial

def train(B=12, debug_overfit=False, oracle_forcing=True):
    def dummy_state_init(rng_key: Key) -> PolicyState:
        return None
    
    canvas_params = CanvasParams()
     # Try to keep same number of examples to fit as in dummier test
    T = canvas_params.max_num_strokes
    H, W = canvas_params.size, canvas_params.size 

    net_key = jax.random.PRNGKey(777) 
    model = BasicCNN(net_key)

    agent_state_init_fn = dummy_state_init
    ref_policy = oracle_policy
    ref_state_init_fn = dummy_state_init

    optim = optax.adamw(1e-3)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    def make_model_dependent_rollout(model, ref_forcing):
        def agent_policy(rng_key: Key, policy_state: PolicyState, env_state: EnvState, observation: FullCanvas, canvas_params: CanvasParams) -> Tuple[PolicyState, Action]:
            return None, model(observation)
        
        def model_dependent_rollout_fn(env_key, pol_key):
            return rollout_trial_batch(env_key, pol_key, B, canvas_params, ref_forcing, agent_policy, agent_state_init_fn, ref_policy, ref_state_init_fn)
        
        return model_dependent_rollout_fn


    @eqx.filter_jit
    def make_step(model, opt_state, env_key, pol_key):
        # Need to compute the actions (inside rollout) within the  value_and_grad'd function, otherwise nothing happens
        # since it makes those static arrays
        def compute_loss(model: BasicCNN):
            # The forward pass MUST happen inside here for JAX to trace it
            rollout_fn = make_model_dependent_rollout(model, oracle_forcing)
            batch_history = rollout_fn(env_key, pol_key)
            preds = batch_history.agent_action
            targets = batch_history.reference_action
            return jnp.mean((preds - targets)**2)

        loss_value, grads = eqx.filter_value_and_grad(compute_loss)(model)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value
    

    def sanity_check_plots(model, batch):
        save_path = Path('tests/oracle_cloning/observation_gifs')
        save_path.mkdir(exist_ok=True, parents=True)
        unforced_rollout_fn = eqx.filter_jit(make_model_dependent_rollout(model, False))
        env_key = jax.random.key(777)
        pol_key = jax.random.key(777)
        batch_history = unforced_rollout_fn(env_key, pol_key)
        obs: Float[Array, "T B 3 H W"] = batch_history.obs
        obs: Float[Array, "B T H W 3"] = obs.transpose(1,0,3,4,2)
        batch_observations = (256* np.asarray(obs, dtype=np.float64)).astype(np.uint8)
        batch_observations = batch_observations
        logging.critical(f"Batch {batch} done in {times[-1]:.5f}s, total rewards: {np.asarray(batch_history.env_state.step_reward, dtype=np.float64).sum()}")
        for b_idx in range(5):
            # TODO: add visualization for the reward
            images = [Image.fromarray(batch_observations[b_idx,t]) for t in range(batch_observations.shape[1])]
            images[0].save(save_path / f'sanity_check_batch_{batch}_trajectory_{b_idx}.gif', save_all=True,
            append_images=images[1:], # append rest of the images
            duration=1000, # in milliseconds
            loop=0)

    
  
    times = []
    losses = [0.] * 10
    for train_step in range(1000):
        env_key = jax.random.key(777+train_step)
        pol_key = jax.random.key(777+train_step)

        if debug_overfit:
            # Forces the same env and oracle stochasticity everytime
            env_key = jax.random.key(777)
            pol_key = jax.random.key(777)

        tic = time.time()
        model, opt_state, loss_value = make_step(model, opt_state, env_key, pol_key)

        times.append(time.time()-tic)
        losses[train_step%10] = loss_value.item()
        if train_step and train_step % 10 == 0:
            print(f"{train_step=}, train_loss={sum(losses):.3e}, time {times[-1]:.3e}")


    for batch in range(5):
        sanity_check_plots(model, batch)
        


if __name__ == '__main__':
    train(debug_overfit=True)