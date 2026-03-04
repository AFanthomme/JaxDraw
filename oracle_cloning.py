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

def train(B=12, debug_overfit=False, oracle_forcing=True):
    def dummy_state_init(key: Key) -> PolicyState:
        return None
    
    canvas_params = CanvasParams()
     # Try to keep same number of examples to fit as in dummier test
    T = canvas_params.max_num_strokes
    H, W = canvas_params.size, canvas_params.size 

    agent_policy = oracle_policy
    agent_state_init_fn = dummy_state_init
    ref_policy = oracle_policy
    ref_state_init_fn = dummy_state_init

    # NOTE: this one does not call net, hence why we jit it separately
    jitted_batch_rollout_fn = jax.jit(
        rollout_trial_batch, 
        static_argnums=(2, 3, 4, 5, 6, 7, 8)
    )


    optim = optax.adamw(1e-3)
    net_key = jax.random.PRNGKey(777) 
    model = BasicCNN(net_key)

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def scan_vmap_model(model: BasicCNN, x: Float[Array, "B T 3 H W"]) -> Tuple[Optional[Float[Array, "B H"]], Float[Array, "B T 3"]]:
        # NOTE: will need slight modification if model gets inputs (x, state) 
        batched_model_forward_fn = jax.vmap(model)

        def time_scan_step_fn(model_state, step_x: Float[Array, "B 3 H W"]):
            return model_state, batched_model_forward_fn(step_x)

        initial_model_state = None
        return jax.lax.scan(time_scan_step_fn, initial_model_state, x)

    @eqx.filter_value_and_grad
    def compute_loss_with_time_dimension(model: BasicCNN, x: Float[Array, "B T 3 H W"], y: Float[Array, "B T 3"]) -> Float[Array, ""]:
        '''
        For now (non-recurrent networks), could do a vmap also across Time axis
        For future-proofing however, introduce a scan already
        '''
        final_state, complete_outputs = scan_vmap_model(model, x)
        return jnp.mean((complete_outputs-y)**2)

    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        loss_value, grads = compute_loss_with_time_dimension(model, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value
    
    # Before this is known to work from basic_net.train_dummy_data
    def agent_state_dummy_init(key: Key) -> PolicyState:
        return None
    
    # Do python for here because this is top-level loop, will need sync anyway to update weights
    times = []
    losses = [0.] * 10
    for train_step in range(5000):
        env_key = jax.random.key(777+train_step)
        pol_key = jax.random.key(777+train_step)

        if debug_overfit:
            # Forces the same env and oracle stochasticity everytime
            env_key = jax.random.key(777)
            pol_key = jax.random.key(777)

        tic = time.time()
        batch_history = jitted_batch_rollout_fn(# Traced
                                        env_key, pol_key, 
                                        # Static
                                        B, canvas_params, oracle_forcing, 
                                        agent_policy, agent_state_init_fn, 
                                        ref_policy, ref_state_init_fn)

        observations: Float[Array, "T B 3 H W"] = batch_history.obs
        target_actions: Float[Array, "T B 3"] = batch_history.reference_action

        # NOTE: for now, we recompute the agent actions separately for descent compared to rollout
        model, opt_state, train_loss = make_step(model, opt_state, observations, target_actions)

        times.append(time.time()-tic)
        losses[train_step%10] = train_loss.item()
        if train_step and train_step % 10 == 0:
            print(f"{train_step=}, train_loss={sum(losses):.3e}, time {times[-1]:.3e}")
        if train_step % 100 in [98, 99]:
            _, pred = scan_vmap_model(model, observations)
            for traj_idx in range(3):
                print(f"traj_idx: {traj_idx} : {pred[traj_idx, :4]}")

if __name__ == '__main__':
    train(debug_overfit=True)