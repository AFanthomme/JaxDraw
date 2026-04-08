import jax
jax.config.update('jax_default_matmul_precision', 'float32')
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from src.custom_types import *
import wandb
from src.ordered_lines_env import on_policy_online_rollout
from src.baseline_policies import baseline_policy_register, make_custom_noise_level_policy
from src.config import EnvParams
from experiments.src.models.observation_policy import ObservationPolicyNetwork
from dataclasses import dataclass
from pathlib import Path
import logging

def main(cfg):
    experiment_output_folder = f"results/02_action_from_obs/{cfg.ruleset}"
    env_params = EnvParams(ruleset=cfg.ruleset, num_target_strokes=cfg.n_lines, max_num_strokes=2*cfg.n_lines+2, size=cfg.env_size, softness=cfg.env_softness, thickness=cfg.env_thickness)
    model = ObservationPolicyNetwork(decoder_config_name=cfg.decoder_config, cnn_config_name=cfg.cnn_config, seed=cfg.model_seed)

    total_steps = cfg.n_epochs * cfg.batches_per_epoch
    schedule = optax.warmup_cosine_decay_schedule(init_value=0., peak_value=cfg.lr, warmup_steps=cfg.warmup_steps, decay_steps=total_steps, end_value=1e-7)
    opt = optax.adamw(learning_rate=schedule, weight_decay=cfg.wd, eps=cfg.adam_eps)

    optim = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip), 
        opt
    )

    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    group_name= f"{cfg.cnn_config}_{cfg.decoder_config}_{cfg.n_lines}lines_{cfg.env_softness}_res_{cfg.env_size}_noise_{cfg.noise_level:.3f}_B_{cfg.batch_size}_{cfg.lr:.2e}_{cfg.wd:.2e}" 
    run_name = f"{group_name}-seed_{cfg.model_seed}"
    out_path = Path(experiment_output_folder) / run_name
    out_path.mkdir(parents=True, exist_ok=True)

    wandb_run = wandb.init(
        project = f"bc_{cfg.ruleset}", 
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

    def l1_loss_fn(model, obs: CanvasHistory, actions: ActionHistory,) -> Float[Array, ""]:
        def scan_fn(carry, inputs):
            step_obs = inputs['obs']
            step_actions = inputs['actions']
            pred = jax.vmap(model)(step_obs)
            return carry, jnp.mean(jnp.abs(pred-step_actions))
        
        initial_carry = jnp.zeros(1)
        inputs = {'obs': obs, 'actions': actions}
        _, output_history = jax.lax.scan(scan_fn, initial_carry, inputs)
        return jnp.mean(output_history)

    def dummy_state_init(rng_key: Key) -> PolicyState:
        return jnp.zeros(1)

    base_policy = baseline_policy_register["ordered_lines_policy"]
    teacher_policy = make_custom_noise_level_policy(base_policy, cfg.noise_level)

    @eqx.filter_jit()
    def learning_step_fn(model, opt_state, env_key, pol_key, do_update=True, return_grads=True):
        # NOTE: since obs is not accessed, it should get optimized away
        rollout: FullRollout = on_policy_online_rollout(# Traced
                        env_key, pol_key, 
                        # Static
                        teacher_policy, dummy_state_init, 
                        cfg.batch_size, env_params, visual=True)
        
        actions = rollout.teacher_action
        obs = rollout.obs

        if do_update or return_grads:
            grad_fn = eqx.filter_value_and_grad(l1_loss_fn)
            loss_value, grads = grad_fn(model, obs, actions)
        else:
            loss_value = l1_loss_fn(model, obs, actions)
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
        if epoch == 0 and step == 0:
            env_key, pol_key = jax.random.split(jax.random.key(0), 2)
            lowered = learning_step_fn.lower(model, opt_state, env_key, pol_key, True, True)

            # Compile it
            compiled = lowered.compile()
            compiled = next(v for v in vars(compiled).values() if hasattr(v, "memory_analysis"))

            # Get the memory analysis dictionary
            analysis = compiled.memory_analysis()

            # This will print the peak temporary memory in bytes
            logging.critical(f"Peak Memory: {analysis.temp_size_in_bytes / (1024**3):.2f} GB")

            # To find the exact heavy tensors, dump the HLO text to a file
            with open(out_path/"hlo_graph.txt", "w") as f:
                f.write(compiled.as_text())

            model, opt_state, train_loss, _ = learning_step_fn(model, opt_state, env_key, pol_key, True, True)
            logging.critical('start the tracing')
            jax.profiler.start_trace(out_path/"jax-trace")
            for b in range(3):
                env_key, pol_key = jax.random.split(jax.random.key(cfg.n_batches_val+step), 2)
                model, opt_state, train_loss, _ = learning_step_fn(model, opt_state, env_key, pol_key, True, True)
                # epoch_losses[b] = train_loss.item()
            jax.block_until_ready(model)
            jax.block_until_ready(opt_state)
            jax.block_until_ready(train_loss)
            jax.profiler.stop_trace()
            logging.critical('done with the tracing')

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
        adam_eps: float = 1e-6
        grad_clip: float = 1.
        n_lines: int = 4

        wd: float = 0.
        env_size: int = 128
        lr: float = 3e-4
        batch_size: int = 128
        cnn_config: str = "small"
        decoder_config: str = 'small'
        noise_level = 1/128
        # noise_level = 0.
        env_softness: float = 2/128
        env_thickness: float = 1/128
        n_epochs: int = 100
        warmup_steps = 100
        n_batches_val: int = 200 
        batches_per_epoch: int = 500
        ruleset = 'along_cardinal_directions'

    run_config = RunConfig()
    main(run_config)

    # for cnn_config, decoder_config in zip(['medium', 'small', ], ['small', 'medium']):
    #     run_config = RunConfig(cnn_config=cnn_config, decoder_config=decoder_config)
    #     main(run_config)


    # for seed in range(4):
    # for decoder_config in ['small', 'no_attention', 'single_layer_mlp']:
    #     # run_config = RunConfig(model_seed=seed, decoder_config=decoder_config)
    #     print(f"Start run with config: \n {run_config}")
    #     main(run_config)