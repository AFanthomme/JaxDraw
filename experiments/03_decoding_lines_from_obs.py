'''
Since it appears MHA can represent the lines -> action part of the policy, let's make sure we can do image -> lines
Use DeTr-inspired training, focus on AdamW with lr schedule and L1 loss since that's what worked best in lines -> action
'''

import jax
jax.config.update('jax_default_matmul_precision', 'float32')
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from src.custom_types import *
import wandb
from src.single_rule_single_trial_env import on_policy_online_rollout
from src.baseline_policies import oracle_policy
from src.config import EnvParams
from src.models.line_set_extractor import LineSetExtractor
from dataclasses import dataclass
from pathlib import Path

experiment_output_folder = "results/03_decoding_lines_from_obs/"

def main(cfg):
    env_params = EnvParams(num_target_strokes=cfg.n_lines, max_num_strokes=2*cfg.n_lines+2, size=cfg.env_size)
    model = LineSetExtractor(num_lines=cfg.n_lines, cnn_config_name=cfg.cnn_config, seed=cfg.model_seed)

    total_steps = cfg.n_epochs * cfg.batches_per_epoch
    schedule = optax.warmup_cosine_decay_schedule(init_value=0., peak_value=cfg.lr, warmup_steps=2000, decay_steps=total_steps, end_value=1e-7)
    opt = optax.adamw(learning_rate=schedule, weight_decay=cfg.wd, eps=cfg.adam_eps)

    optim = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip), 
        opt
    )

    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    group_name= f'{cfg.n_lines}_lines_res_{cfg.env_size}_{cfg.cnn_config}_{cfg.lr:.2e}_{cfg.wd:.2e}' 
    run_name = f"{group_name}-seed_{cfg.model_seed}"
    out_path = Path(experiment_output_folder) / run_name
    out_path.mkdir(parents=True, exist_ok=True)

    wandb_run = wandb.init(
        project = "lines_from_obs", 
        dir=experiment_output_folder,
        group = group_name,
        name = run_name,
        tags=["dummy",],
        config=run_config.__dict__,
        save_code=True 
    )
    wandb.define_metric("epoch")
    wandb.define_metric("Epoch/*", step_metric="epoch")
    wandb.define_metric("Histograms/*", step_metric="epoch")

    @jax.vmap
    def hungarian_l1_loss(pred_lines: chex.Array, target_lines: chex.Array) -> chex.Array:
        # Removing "done" from the matching procedure, could be bad during initial steps
        cost_matrix = jnp.sum(jnp.abs(pred_lines[:, None, :4] - target_lines[None, :, :4]), axis=-1)
        row_idx, col_idx = optax.assignment.hungarian_algorithm(cost_matrix)
        loss = jnp.mean(jnp.abs(pred_lines[row_idx] - target_lines[col_idx]))
        return loss

    def loss_fn(model, obs: CanvasHistory, strokes: TargetStrokesHistory, statuses: TargetStrokesStatusHistory, positions: CoordinateHistory) -> Float[Array, ""]:
        def scan_fn(carry, inputs):
            # Note those are batches, so T leading dim is gone
            step_obs = inputs['obs']
            step_strokes = inputs['strokes']
            step_statuses = inputs['statuses']
            step_positions = inputs['positions']
            broadacasted_pos = jnp.repeat(step_positions[:, None, :], cfg.n_lines, axis=1)
            target = jnp.concatenate([step_strokes, step_statuses[:,:,None], broadacasted_pos], axis=-1)
            pred = jax.vmap(model)(step_obs)[0]
            step_loss = hungarian_l1_loss(pred, target)

            # target = jnp.concatenate([step_strokes, step_statuses[:,:,None]], axis=-1)
            # pred_lines, pred_agent_pos = jax.vmap(model)(step_obs)
            # lines_loss = hungarian_l1_loss(pred_lines, target)
            # agent_loss =  jnp.mean(jnp.abs(pred_agent_pos - step_positions))
            # step_loss = (n_lines * lines_loss + agent_loss) /(n_lines+1)
            return carry, step_loss

        initial_carry = jnp.zeros(1)
        inputs = {'strokes': strokes, 'statuses': statuses, 'obs': obs, 'positions': positions}
        _, output_history = jax.lax.scan(scan_fn, initial_carry, inputs)
        return jnp.mean(output_history)

    def dummy_state_init(rng_key: Key) -> PolicyState:
        return jnp.zeros(1)

    @eqx.filter_jit()
    def learning_step_fn(model, opt_state, env_key, pol_key, do_update=True, return_grads=True):
        # NOTE: since obs is not accessed, it should get optimized away
        rollout: FullRollout = on_policy_online_rollout(# Traced
                        env_key, pol_key, 
                        # Static
                        oracle_policy, dummy_state_init, 
                        cfg.batch_size, env_params, visual=True)
        
        state_history = rollout.env_state
        strokes = state_history.target_strokes
        strokes_status = state_history.target_strokes_status
        positions = state_history.position
        obs = rollout.obs

        if do_update or return_grads:
            grad_fn = eqx.filter_value_and_grad(loss_fn)
            loss_value, grads = grad_fn(model, obs, strokes, strokes_status, positions)
        else:
            loss_value = loss_fn(model, obs, strokes, strokes_status, positions)
            grads = None

        if do_update:
            updates, opt_state = optim.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            model = eqx.apply_updates(model, updates)

        return model, opt_state, loss_value, grads

    step = 0
    for epoch in range(cfg.n_epochs):
        epoch_losses = np.zeros(cfg.batches_per_epoch)

        for b in range(cfg.batches_per_epoch):
            step +=1
            env_key, pol_key = jax.random.split(jax.random.key(cfg.n_batches_val+step), 2)
            model, opt_state, train_loss, _ = learning_step_fn(model, opt_state, env_key, pol_key, True, True)
            epoch_losses[b] = train_loss.item()

            metrics = {'Step/Train loss (log10)': np.log10(train_loss.item()+1e-20),}
            if b != cfg.batches_per_epoch -1:
                wandb.log(metrics, step=step)

        # Validation
        model = eqx.nn.inference_mode(model, value=True)
        val_losses = np.zeros(cfg.n_batches_val)
        for val_batch_idx in range(cfg.n_batches_val):
            env_key, pol_key = jax.random.split(jax.random.key(val_batch_idx), 2)
            _, _, val_loss, _ = learning_step_fn(model, opt_state, env_key, pol_key, False, False)
            val_losses[val_batch_idx] = val_loss.item()
        model = eqx.nn.inference_mode(model, value=False)

        # save model, need refinements
        eqx.tree_serialise_leaves(out_path / f"model_{step}.eqx", model)

        print(f"Epoch {epoch}: train loss : {np.mean(epoch_losses):.3e}, val loss {np.mean(val_losses):.3e}")
        metrics.update({'Epoch/Val loss (log10)': np.mean(np.log10(val_losses+1e-20))})
        metrics.update({'Histograms/Val loss  (log10)': wandb.Histogram([i for i in np.log10(val_losses+1e-20)])})
        metrics.update({'Epoch/Train loss (log10)': np.mean(np.log10(epoch_losses+1e-20))})
        metrics.update({'Histograms/Train loss (log10)': wandb.Histogram([i for i in np.log10(epoch_losses+1e-20)])})
        metrics.update({'epoch': epoch})
        wandb.log(metrics, step=step)

    wandb_run.finish()

    return model


if __name__ == '__main__':
    @dataclass
    class RunConfig:
        model_seed: int = 0
        batch_size: int = 16
        n_epochs: int = 100
        adam_eps: float = 1e-6
        lr: float = 1e-4
        wd: float = 1e-6
        grad_clip: float = 1.
        batches_per_epoch: int = 500
        n_batches_val: int = 50
        n_lines: int = 4
        cnn_config: str = "small"
        env_size: int = 64

    run_config = RunConfig()
    print(f"Start run with config: \n {run_config}")
    main(run_config)


    # config = RunConfig()
    # print(f"Start run with config: \n {config}")
    # overfit_oracle(config) 