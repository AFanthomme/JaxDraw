import jax
import jax.numpy as jnp
import equinox as eqx
import chex
from .convolutional import build_convnet
from .perception_decoder import build_decoder


class LineSetExtractor(eqx.Module):
    cnn: eqx.Module
    decoder: eqx.Module
    proj: eqx.nn.Conv2d
    norm: eqx.nn.LayerNorm

    def __init__(self, num_lines: int, cnn_config_name: str, seed: int = 777):
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
        # query_groups = ((num_lines, 5), (1, 2)) # x1,x2,y1,y2,line_done for each line; agent_x,agent_y 
        query_groups = ((num_lines, 7), ) # x1,x2,y1,y2,line_done,agent_x,agent_y 

        self.cnn = build_convnet(config_name=cnn_config_name, seed=int(k1[0]))
        self.decoder = build_decoder(query_groups, seed=int(k2[0]))
        
        cnn_out_channels = self.cnn.out_channels

        self.proj = eqx.nn.Conv2d(
            in_channels=cnn_out_channels + 2, 
            out_channels=self.decoder.embed_dim, 
            kernel_size=1, 
            key=k3
        )

        self.norm = eqx.nn.LayerNorm(self.decoder.embed_dim)

    def _append_coords(self, feature_map: chex.Array) -> chex.Array:
        # Similar to "CoordConv", used to add spatial context to the tokens 
        _, h, w = feature_map.shape
        y_coords = jnp.linspace(-1, 1, h)
        x_coords = jnp.linspace(-1, 1, w)
        y_grid, x_grid = jnp.meshgrid(y_coords, x_coords, indexing='ij')
        coords = jnp.stack([y_grid, x_grid], axis=0)
        return jnp.concatenate([feature_map, coords], axis=0)

    def __call__(self, image: chex.Array) -> chex.Array:
        cnn_features = self.cnn(image)
        cnn_with_coords = self._append_coords(cnn_features)
        projected_features = self.proj(cnn_with_coords)
        c, h, w = projected_features.shape
        flattened = projected_features.reshape(c, h * w)
        memory = flattened.T
        normed_memory = jax.vmap(self.norm)(memory)
        return self.decoder(normed_memory)
