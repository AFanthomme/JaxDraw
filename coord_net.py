# # This is probably not needed but tried during debugging 
# class CoordConv2d(eqx.nn.Conv2d):
#     """
#     Subclass of eqx.nn.Conv2d that prepends normalized (x, y) 
#     coordinates to the input channels.
#     """

#     def __init__(self, in_channels: int, *args, **kwargs):
#         # We add 2 to in_channels to account for the x and y coordinate maps
#         super().__init__(in_channels + 2, *args, **kwargs)

#     def __call__(self, x: jax.Array) -> jax.Array:
#         """
#         Input x: (in_channels, H, W)
#         Output: (out_channels, H_out, W_out)
#         """
#         # 1. Get dimensions
#         c, h, w = x.shape
        
#         # 2. Create coordinate grids (0 to H-1, 0 to W-1)
#         # Using linspace to get normalized coordinates [-1, 1]
#         # This helps with gradient stability
#         y_coords = jnp.linspace(-1, 1, h)
#         x_coords = jnp.linspace(-1, 1, w)
        
#         # 3. Create the 2D meshgrid
#         # Resulting shapes: (H, W)
#         y_grid, x_grid = jnp.meshgrid(y_coords, x_coords, indexing='ij')
        
#         # 4. Add channel dimension: (1, H, W)
#         y_grid = jnp.expand_dims(y_grid, axis=0)
#         x_grid = jnp.expand_dims(x_grid, axis=0)
        
#         # 5. Concatenate along the channel axis
#         # New shape: (in_channels + 2, H, W)
#         x_augmented = jnp.concatenate([x, y_grid, x_grid], axis=0)
        
#         # 6. Pass to the original Conv2d logic
#         return super().__call__(x_augmented)

# class BasicCNN(eqx.Module):
#     '''
#     Start with LayerNorm since it's batch size independant, and could behave better in RNN cases but needs to be investigated
#     Use group convolutions, but mostly for fun; LN after activation, once again quite debatable and probably insignificant
#     '''
#     layers: list

#     def __init__(self, key):
#         layer_keys = jax.random.split(key, 7)

#         # Remove 2 to each output filter number since we'll add them back before next layer via meshgrid
#         self.layers = [
#             CoordConv2d(3, 126, kernel_size=5, stride=2, key=layer_keys[0]),
#             jax.nn.relu, 
#             eqx.nn.LayerNorm([126, 62, 62]),
#             CoordConv2d(126, 254, kernel_size=5, stride=2, key=layer_keys[1]),
#             jax.nn.relu, 
#             eqx.nn.LayerNorm([254, 29, 29]),
#             CoordConv2d(254, 254, kernel_size=3, stride=2, key=layer_keys[2]),
#             jax.nn.relu, 
#             eqx.nn.LayerNorm([254, 14, 14]),
#             CoordConv2d(254, 126, kernel_size=3, stride=2, key=layer_keys[3]),
#             jax.nn.relu, 
#             eqx.nn.LayerNorm([126, 6, 6]),
#             CoordConv2d(126, 64, kernel_size=3, stride=2, key=layer_keys[4]),
#             jax.nn.relu, 
#             eqx.nn.LayerNorm([64, 2, 2]),
#             jnp.ravel,
#             eqx.nn.Linear(256, 3, key=layer_keys[5], use_bias=False),
#             jax.nn.tanh # Force network output in reasonable range
#         ]

#     def __call__(self, x: FullCanvas) -> PolicyNetworkOutput:
#         for layer in self.layers:
#             x = layer(x)
#         return x
