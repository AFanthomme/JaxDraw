import logging
from pathlib import Path
from dataclasses import dataclass
from src.image_utils import plot_cumulated_rewards
from src.eval_utils import sanity_check, compute_running_statistics
from src.models.envstate_policy import envstate_rule_based_policy_config_register, EnvStateRuleBasedPolicyNetwork, EnvStateRuleBasedPolicy
from src.config import EnvParams
from src.baseline_policies import baseline_policy_register, make_custom_noise_level_policy
from src.environment import on_policy_online_rollout
from typing import cast
from src.custom_types import *
import wandb
import optax
import equinox as eqx
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update('jax_default_matmul_precision', 'float32')


def main(cfg):
    experiment_output_folder = f"results/envstate_BC/{cfg.ruleset}"
    group_name = f"{cfg.model_config}_B_{cfg.batch_size}"
    run_name = f"{group_name}-seed_{cfg.model_seed}"
    out_path = Path(experiment_output_folder) / run_name
    vis_path = Path(experiment_output_folder) / run_name / "visualizations"
    vis_path.mkdir(exist_ok=True, parents=True)
    out_path.mkdir(parents=True, exist_ok=True)

    env_params = EnvParams(ruleset=cfg.ruleset, num_target_strokes=cfg.n_lines, max_num_strokes=2 *
                           cfg.n_lines+2, size=cfg.env_size, softness=cfg.env_softness, thickness=cfg.env_thickness)
    model = EnvStateRuleBasedPolicyNetwork(
        envstate_rule_based_policy_config_register[cfg.model_config], jax.random.key(cfg.model_seed))

    schedule = optax.schedules.sgdr_schedule(
                    [dict(
                            init_value=0.,
                            peak_value=cfg.cycle_start_lrs[i],
                            end_value=cfg.cycle_end_lrs[i],
                            decay_steps=cfg.cycle_duration_in_epochs[i]*cfg.batches_per_epoch,
                            warmup_steps=cfg.warmup_duration_in_epochs[i]*cfg.batches_per_epoch,
                        ) 
                    for i in range(len(cfg.cycle_start_lrs)) 
                    ]
                )
 
    opt = optax.adamw(learning_rate=schedule,
                      weight_decay=cfg.wd, eps=cfg.adam_eps)

    optim = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        opt
    )

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    wandb_run = wandb.init(
        project=f"envstate_BC_{cfg.ruleset}",
        dir=experiment_output_folder,
        group=group_name,
        name=run_name,
        tags=["dummy",],
        config=cfg.__dict__,
        save_code=True
    )
    wandb.define_metric("epoch")
    wandb.define_metric("Epoch/*", step_metric="epoch")
    wandb.define_metric("Histograms/*", step_metric="epoch")

    def loss_fn(model: EnvStateRuleBasedPolicyNetwork, env_states: EnvStateHistory, target_actions: ActionHistory) -> Float[Array, ""]:
        def scan_fn(carry: jax.Array, inputs: Tuple[EnvStateBatch, ActionBatch]):
            step_envstates, step_actions = inputs
            pred = jax.vmap(model)(cast(EnvState, step_envstates))
            return carry, jnp.mean(jnp.abs(pred-step_actions))

        initial_carry = jnp.zeros(1)
        _, output_history = jax.lax.scan(
            scan_fn, initial_carry, (cast(EnvStateBatch, env_states), cast(ActionBatch, target_actions)))
        return jnp.mean(output_history)

    def dummy_state_init(rng_key: Key) -> PolicyState:
        return jnp.zeros(1)

    base_policy = baseline_policy_register["ordered"]
    teacher_policy = make_custom_noise_level_policy(
        base_policy, cfg.noise_level)

    @eqx.filter_jit()
    def learning_step_fn(model, opt_state, env_key, pol_key, do_update=True, return_grads=True):
        # NOTE: since obs is not accessed, it should get optimized away
        rollout: FullRollout = on_policy_online_rollout(  # Traced
            env_key, pol_key,
            # Static
            teacher_policy, dummy_state_init,
            cfg.batch_size, env_params, visual=False)

        state_history = rollout.env_state
        target_actions = rollout.teacher_action

        if do_update or return_grads:
            grad_fn = eqx.filter_value_and_grad(loss_fn)
            loss_value, grads = grad_fn(model, state_history, target_actions)
        else:
            loss_value = loss_fn(model, state_history, target_actions)
            grads = None

        if do_update:
            assert grads is not None
            updates, opt_state = optim.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            model = eqx.apply_updates(model, updates)

        return model, opt_state, loss_value, grads

    step = 0
    metrics = {}
    plot_data = {}
    plot_colors = {
                'network': 'blue',
                'closest': 'orange',
                'ordered': 'darkgreen',
            }
    for pol_name in plot_colors.keys():
        if pol_name == 'network':
            continue
        policy = baseline_policy_register[pol_name]
        logging.critical(
            f"Starting test rollouts for agent_policy {pol_name}")
        plot_data[pol_name] = compute_running_statistics(
            env_params, policy, dummy_state_init, n_batches=cfg.n_batches_val, batch_size=cfg.batch_size)


    for epoch in range(cfg.n_epochs):
        epoch_losses = np.zeros(cfg.batches_per_epoch)

        if epoch == 0 and step == 0:
            logging.critical(f"Executing pre-run profiling")
            env_key, pol_key = jax.random.split(jax.random.key(0), 2)
            lowered = learning_step_fn.lower(model, opt_state, env_key, pol_key, True, True)
            compiled = lowered.compile()
            compiled = next(v for v in vars(compiled).values()
                            if hasattr(v, "memory_analysis"))
            analysis = compiled.memory_analysis()
            logging.critical(
                f"Peak Memory: {analysis.temp_size_in_bytes / (1024**3):.2f} GB")
            with open(out_path/"hlo_graph.txt", "w") as f:
                f.write(compiled.as_text())
            model, opt_state, train_loss, _ = learning_step_fn(
                model, opt_state, env_key, pol_key, True, True)
            logging.critical('start the tracing')
            jax.profiler.start_trace(out_path/"jax-trace")
            for b in range(3):
                env_key, pol_key = jax.random.split(
                    jax.random.key(cfg.n_batches_val+step), 2)
                model, opt_state, train_loss, _ = learning_step_fn(
                    model, opt_state, env_key, pol_key, True, True)
            jax.block_until_ready(model)
            jax.block_until_ready(opt_state)
            jax.block_until_ready(train_loss)
            jax.profiler.stop_trace()
            logging.critical('Done with the pre-run profiling')

        for b in range(cfg.batches_per_epoch):
            step += 1
            env_key, pol_key = jax.random.split(
                jax.random.key(cfg.n_batches_val+step), 2)
            model, opt_state, train_loss, _ = learning_step_fn(
                model, opt_state, env_key, pol_key, True, True)
            epoch_losses[b] = train_loss.item()

            metrics = {
                'Step/Train loss (log10)': np.log10(train_loss.item()+1e-20), }
            if b != cfg.batches_per_epoch - 1:
                wandb.log(metrics, step=step)

        # Validation
        model = eqx.nn.inference_mode(model, value=True)
        val_losses = np.zeros(cfg.n_batches_val)
        for val_batch_idx in range(cfg.n_batches_val):
            env_key, pol_key = jax.random.split(
                jax.random.key(val_batch_idx), 2)
            _, _, val_loss, _ = learning_step_fn(
                model, opt_state, env_key, pol_key, False, False)
            val_losses[val_batch_idx] = val_loss.item()

        # save model, need refinements
        eqx.tree_serialise_leaves(out_path / f"model_{step}.eqx", model)

        print(
            f"Epoch {epoch}: train loss : {np.mean(epoch_losses):.3e}, val loss {np.mean(val_losses):.3e}")
        metrics.update(
            {'Epoch/Val loss (log10)': np.mean(np.log10(val_losses+1e-20))})
        metrics.update(
            {'Histograms/Val loss  (log10)': wandb.Histogram([i for i in np.log10(val_losses+1e-20)])})
        metrics.update(
            {'Epoch/Train loss (log10)': np.mean(np.log10(epoch_losses+1e-20))})
        metrics.update(
            {'Histograms/Train loss (log10)': wandb.Histogram([i for i in np.log10(epoch_losses+1e-20)])})
        metrics.update({'epoch': epoch})
        wandb.log(metrics, step=step)

        if epoch % 10 == 0 or epoch == cfg.n_epochs - 1:
            agent_policy = EnvStateRuleBasedPolicy(model)
            plot_data['network'] = compute_running_statistics(
                env_params, agent_policy, dummy_state_init, n_batches=cfg.n_batches_val, batch_size=cfg.batch_size)
            plot_cumulated_rewards(
                plot_data, plot_colors, env_params, vis_path, f"_{step}")
            sanity_check(env_params, agent_policy, dummy_state_init,
                         'Agent network policy', vis_path)
        model = eqx.nn.inference_mode(model, value=False)

    wandb_run.finish()

    return model


if __name__ == '__main__':
    @dataclass
    class RunConfig:
        model_seed: int = 0
        n_lines: int = 5
        env_size: int = 128
        noise_level = 0.
        env_softness: float = 2/128
        env_thickness: float = 1/128
        adam_eps: float = 1e-6
        wd: float = 0.
        optimizer: str = 'adamw'
        grad_clip: float = 1.
        n_epochs: int = 100
        batch_size: int = 512
        batches_per_epoch: int = 8000
        n_batches_val: int = 400
        model_config: str = 'small'
        # ruleset = 'parametric_directions_with_decreasing'
        # ruleset = 'modes_zero_to_three'
        ruleset = 'any'
        # cosine anneal with warm restarts
        cycle_start_lrs = [1e-3,]
        cycle_end_lrs = [1e-6,]
        cycle_duration_in_epochs = [100]
        warmup_duration_in_epochs = [3., 3.]


    run_config = RunConfig()
    print(f"Start run with config: \n {run_config}")
    main(run_config)
