import jax
import jax.numpy as jnp
import equinox as eqx
from src.custom_types import *

class CoordConv2d(eqx.nn.Conv2d):
    """
    Subclass of eqx.nn.Conv2d that prepends normalized (x, y) coordinates to the input channels.
    Apparently can help in visual regression tasks (avoids having to embed coordinate map in weights,
    saving more training gradient and representation power for more interesting stuff)
    """

    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs):
        # Need to add 2 to in_channels for coordinate maps, rest kept identical
        super().__init__(in_channels + 2, out_channels, *args, **kwargs)

    def __call__(self, x: Float[Array, "C H W"]) -> Float[Array, "C_out H_out W_out"]:
        C, H, W = x.shape
        y_coords = jnp.linspace(-1, 1, H)
        x_coords = jnp.linspace(-1, 1, W)
        y_grid, x_grid = jnp.meshgrid(y_coords, x_coords, indexing='ij')
        y_grid: Float[Array, "1 H W"] = jnp.expand_dims(y_grid, axis=0)
        x_grid: Float[Array, "1 H W"] = jnp.expand_dims(x_grid, axis=0)
        x_augmented = jnp.concatenate([x, y_grid, x_grid], axis=0)
        return super().__call__(x_augmented)

class IdentityLayer(eqx.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x: Array) -> Array:
        return x

class BasicCNN(eqx.Module):
    '''
    Start with LayerNorm since it's batch size independent, stateless, and might behave better in RNN
    LN after activation, once again quite debatable and probably insignificant

    Spatial shapes, strides and kernel sizes hardcoded because of LayerNorm requirements
    '''
    layers: list

    def __init__(self, key, use_coord_conv:bool=False, use_layernorm:bool=True, out_scale:float=1.):
        layer_keys = jax.random.split(key, 8)
        n_channels_conv = [128, 256, 256, 128, 64]
        n_channels_mlp = [256, 256, 128]


        conv_factory = lambda *args, **kwargs: eqx.nn.Conv2d(*args, **kwargs)
        norm_factory = lambda *args, **kwargs: IdentityLayer(*args, **kwargs)
        # if use_coord_conv:
        #     conv_factory = lambda *args, **kwargs: CoordConv2d.__init__(*args, **kwargs)
        #     # Remove 2 so final shapes remain nice powers of 2 after adding coordinates, not used for output
        #     n_channels_conv = [n-2 for i, n in enumerate(n_channels_conv) if i != len(n_channels_conv) - 1]
        # else:
        #     conv_factory = eqx.nn.Conv2d.__init__

        # if use_layernorm:
        #     norm_factory = eqx.nn.LayerNorm
        # else:
        #     identity_layer = lambda y : y
        #     norm_factory = lambda x: identity_layer

        def scaled_tanh(x):
            return out_scale * jax.nn.tanh(x)
        
        self.layers = [
            conv_factory(3, n_channels_conv[0], kernel_size=5, stride=2, key=layer_keys[0]),
            # norm_factory([n_channels_conv[0], 62, 62]),
            jax.nn.relu, 
            conv_factory(n_channels_conv[0], n_channels_conv[1], kernel_size=5, stride=2, key=layer_keys[1]),
            # norm_factory([n_channels_conv[1], 29, 29]),
            jax.nn.relu, 
            conv_factory(n_channels_conv[1], n_channels_conv[2], kernel_size=3, stride=2, key=layer_keys[2]),
            # norm_factory([n_channels_conv[2], 14, 14]),
            jax.nn.relu, 
            conv_factory(n_channels_conv[2], n_channels_conv[3], kernel_size=3, stride=2, key=layer_keys[3]),
            # norm_factory([n_channels_conv[3], 6, 6]),
            jax.nn.relu, 
            conv_factory(n_channels_conv[3], n_channels_conv[4], kernel_size=3, stride=2, key=layer_keys[4]),
            # norm_factory([n_channels_conv[4], 2, 2]),
            jax.nn.relu, 
            jnp.ravel,
            eqx.nn.Linear(n_channels_mlp[0], n_channels_mlp[1], key=layer_keys[5]),
            jax.nn.relu, 
            eqx.nn.Linear(n_channels_mlp[1], n_channels_mlp[2], key=layer_keys[6]),
            jax.nn.relu, 
            eqx.nn.Linear(n_channels_mlp[2], 3, key=layer_keys[7]),
            # Force network output in reasonable range, could lead to null grad but need to clip one way or another
            scaled_tanh
        ]

    def __call__(self, x: FullCanvas) -> Action:
        for layer in self.layers:
            x = layer(x)
        return x