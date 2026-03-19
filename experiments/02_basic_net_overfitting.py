import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from src.basic_net import BasicCNN
from src.net_utils import log_model_health
from src.custom_types import *
import wandb
from copy import deepcopy

experiment_output_folder = "results/02_basic_nets/"

base_config = {
    'model_seed': 0, 
    'use_coord_conv': False, 
    'n_batches': 1, 
    'n_steps': 1000, 
    'data_seed': 777,
    'model_seed': 0,
    'batch_size': 128
}

def train_dummy_data(config):
    B = config['batch_size']
    n_steps = config['n_steps']
    use_coord_conv = config['use_coord_conv']
    n_batches = config['n_batches']
    model_seed = config['model_seed']
    data_seed = config['data_seed']

    model = BasicCNN(jax.random.PRNGKey(data_seed + model_seed), use_coord_conv=use_coord_conv)


    # Do half of the steps at 1e-3, then switch to 1e-4 to get lower, and finish with some polish at 5e-5
    lr_schedule = optax.schedules.piecewise_constant_schedule(init_value=1e-3, boundaries_and_scales={n_steps // 2: .1, (3*n_steps)//4: .5})
    optim = optax.adamw(lr_schedule)

    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    group_name=f'n_batches_{n_batches}_coordconv_{use_coord_conv}' 

    wandb.init(
        project = "basic_overfitting_dummy_data", 
        dir=experiment_output_folder,
        group = group_name,
        name = f"{group_name}-seed_{model_seed}",
        tags=["dummy",],
        config=config,
        save_code=True 
    )


    @eqx.filter_value_and_grad
    def loss_value_and_grad_fn(model: BasicCNN, x, y) -> Float[Array, ""]:
        pred = jax.vmap(model)(x)
        return jnp.mean((pred-y)**2)

    @eqx.filter_jit
    def train_step(model, opt_state, x, y):
        loss_value, grads = loss_value_and_grad_fn(model, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value
    
    @eqx.filter_jit
    def test_step(model, x, y):
        loss_value, grads = loss_value_and_grad_fn(model, x, y)
        return loss_value, grads

    for step in range(n_steps):
        # NOTE: obs, target will aternate between n_batches values
        # We regenerate every step becaus enot limiting and not the point of this particular study
        obs_key, target_key = jax.random.split(jax.random.PRNGKey(data_seed+step%n_batches), 2)
        obs: FullCanvas = jax.random.normal(obs_key, shape=(B,3,128,128))
        target: Action = jax.random.uniform(target_key, shape=(B,3,), minval=-1, maxval=1)

        model, opt_state, train_loss = train_step(model, opt_state, obs, target)

        wandb.log({'Train loss (log10)': jnp.log10(train_loss.item()+1e-10)}, step=step)

        if not step % 10:
            print(f"Step {step}: loss : {train_loss.item():.3e}")

        if not step % 100:
            _, grads = test_step(model, obs, target) 
            log_model_health(model, grads, step)
    return model



if __name__ == '__main__':
    # NOTE: since the data is completely arbitrary, should not expect any generalization
    # Hence, need to increase number of steps proportionally to n_batches, and will reach 
    # network capacity saturation at some point; give more steps to small n_batches just 
    # to help with warmup
    for seed in range(4):
        for use_coord_conv in [True, False]:
            config = deepcopy(base_config)
            config['model_seed'] = seed
        train_dummy_data(config)
        # train_dummy_data(n_batches=10, n_steps=2000)
        # train_dummy_data(n_batches=100, n_steps=5000)