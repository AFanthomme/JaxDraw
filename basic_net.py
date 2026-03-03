import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from custom_types import *

class BasicCNN(eqx.Module):
    '''
    Start with LayerNorm since it's batch size independant, and could behave better in RNN cases but needs to be investigated
    Use group convolutions, but mostly for fun; LN after activation, once again quite debatable and probably insignificant
    '''
    layers: list

    def __init__(self, key):
        layer_keys = jax.random.split(key, 7)

        self.layers = [
            eqx.nn.Conv2d(3, 128, kernel_size=5, stride=2, key=layer_keys[0]),
            eqx.nn.LayerNorm([128, 62, 62]),
            jax.nn.relu, 
            eqx.nn.Conv2d(128, 256, kernel_size=5, stride=2, key=layer_keys[1]),
            eqx.nn.LayerNorm([256, 29, 29]),
            jax.nn.relu, 
            eqx.nn.Conv2d(256, 256, kernel_size=3, stride=2, key=layer_keys[2]),
            eqx.nn.LayerNorm([256, 14, 14]),
            jax.nn.relu, 
            eqx.nn.Conv2d(256, 128, kernel_size=3, stride=2, key=layer_keys[3]),
            eqx.nn.LayerNorm([128, 6, 6]),
            jax.nn.relu, 
            eqx.nn.Conv2d(128, 64, kernel_size=3, stride=2, key=layer_keys[4]),
            eqx.nn.LayerNorm([64, 2, 2]),
            jax.nn.relu, 
            jnp.ravel,
            eqx.nn.Linear(256, 3, key=layer_keys[5]),
            # Force network output in reasonable range, could lead to null grad but need to clip one way or another
            jax.nn.tanh 
        ]

    def __call__(self, x: FullCanvas) -> Action:
        for layer in self.layers:
            x = layer(x)
        return x

# This works, so the problem in behavior_cloning must come from somewhere else
def train_dummy_data():
    B = 8    
    optim = optax.adamw(5e-3)
    net_key = jax.random.PRNGKey(777) 
    model = BasicCNN(net_key)

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_value_and_grad
    def loss(model: BasicCNN, x, y) -> Float[Array, ""]:
        pred = jax.vmap(model)(x)
        return jnp.mean((pred-y)**2)

    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        loss_value, grads = loss(model, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value
    
    for step in range(1000):
        # NOTE: obs, target should aternate between two values
        
        obs_key, target_key = jax.random.split(jax.random.PRNGKey(777+step%2), 2)
        obs: FullCanvas = jax.random.normal(obs_key, shape=(B,3,128,128))
        target: Action = jax.random.normal(target_key, shape=(B,3,))

        model, opt_state, train_loss = make_step(model, opt_state, obs, target)
        print(f"{step=}, train_loss={train_loss.item()}")
        if step % 100 in [98, 99]:
            pred=jax.vmap(model)(obs)
            for traj_idx in range(3):
                print(f"traj_idx: {traj_idx} : {pred[traj_idx]}")
    return model



if __name__ == '__main__':
    train_dummy_data()