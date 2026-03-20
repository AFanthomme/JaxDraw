import jax
jax.config.update('jax_default_matmul_precision', 'float32')
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from src.basic_net import BasicCNN
from src.net_utils import log_model_health
from src.custom_types import *
import wandb
from copy import deepcopy
from src.single_rule_single_trial_env import on_policy_online_rollout
from src.baseline_policies import oracle_policy
from src.config import EnvParams

experiment_output_folder = "results/03_oracle_cloning/"

base_config = {
    'model_seed': 0, 
    'use_coord_conv': False, 
    'n_batches': 1, 
    'n_epochs': 100, 
    'data_seed': 777,
    'model_seed': 0,
    'batch_size': 8,
    'use_layernorm': True,
    'weight_decay': 0.,
    'adam_eps': 1e-6
}

def overfit_oracle(config):
    '''
    We use the online rollout within the training loop to profile its impact
    '''
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    use_coord_conv = config['use_coord_conv']
    n_batches = config['n_batches']
    model_seed = config['model_seed']
    data_seed = config['data_seed']
    use_layernorm = config['use_layernorm']
    weight_decay = config['weight_decay']
    adam_eps = config['adam_eps']

    model = BasicCNN(jax.random.key(data_seed + model_seed), use_coord_conv=use_coord_conv, use_layernorm=use_layernorm)

    lr_schedule = optax.schedules.piecewise_constant_schedule(init_value=2e-4, boundaries_and_scales={(3*n_epochs)//4: .3})
    optim = optax.adamw(lr_schedule, weight_decay=weight_decay, eps=adam_eps)


    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    group_name= f'n_batches_{n_batches}_coordconv_{use_coord_conv}_layernorm_{use_layernorm}' 

    wandb_run = wandb.init(
        project = "overfit_cloning", 
        dir=experiment_output_folder,
        group = group_name,
        name = f"{group_name}-seed_{model_seed}",
        tags=["dummy",],
        config=config,
        save_code=True 
    )

    @eqx.filter_value_and_grad
    def loss_value_and_grad_fn(model: BasicCNN, x: CanvasHistory, y: ActionHistory) -> Float[Array, ""]:
        def scan_fn(carry, inputs):
            x = inputs['x']
            y = inputs['y']
            pred = jax.vmap(model)(x)
            return carry, jnp.mean((pred-y)**2)
        initial_carry = jnp.zeros(1)
        inputs = {'x': x, 'y': y}
        _, output_history = jax.lax.scan(scan_fn, initial_carry, inputs)
        return jnp.mean(output_history)

    env_params = EnvParams()
    def dummy_state_init(rng_key: Key) -> PolicyState:
        return jnp.zeros(1)

    @eqx.filter_jit
    def train_step(model, opt_state, env_key, pol_key):
        batch_rollout: FullRollout = on_policy_online_rollout(# Traced
                        env_key, pol_key, 
                        # Static
                        oracle_policy, dummy_state_init, 
                        batch_size, env_params)
        
        x = batch_rollout.obs
        y = batch_rollout.teacher_action

        loss_value, grads = loss_value_and_grad_fn(model, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value
    
    # @eqx.filter_jit
    # def test_step(model, x, y):
    #     loss_value, grads = loss_value_and_grad_fn(model, x, y)
    #     return loss_value, grads


    for epoch in range(n_epochs):
        epoch_losses = np.zeros(n_batches)
        epoch_key = jax.random.key(epoch)
        perm = jax.random.permutation(epoch_key, n_batches)

        for b in range(n_batches):
            env_key, pol_key = jax.random.split(jax.random.key(data_seed+perm[b]), 2)
            model, opt_state, train_loss = train_step(model, opt_state, env_key, pol_key)
            epoch_losses[b] = train_loss.item()

        print(f"Epoch {epoch}: loss : {np.mean(epoch_losses):.3e}")

        wandb.log({'Train loss (log10)': np.log10(np.mean(epoch_losses)+1e-20)}, step=(epoch+1)*n_batches)

    wandb_run.finish()

    return model



if __name__ == '__main__':
    config = deepcopy(base_config)
    config['model_seed'] = 0
    config['use_coord_conv'] = True
    config['use_layernorm'] = True
    config['n_batches'] = 10
    config['n_epochs'] = 1000
    print(f"Start run with config: \n {config}")
    overfit_oracle(config)

    # for use_layernorm in [True, False, ]:
    #     for seed in range(4):
    #         for use_coord_conv in [True, False]:
    #             config = deepcopy(base_config)
    #             config['n_batches'] = 5
    #             config['n_epochs'] = 1000
    #             config['model_seed'] = seed
    #             config['use_coord_conv'] = use_coord_conv
    #             config['use_layernorm'] = use_layernorm
    #             print(f"Start run with config: \n {config}")

    #             overfit_oracle(config)
