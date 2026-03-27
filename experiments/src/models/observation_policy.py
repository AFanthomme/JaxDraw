
import jax
import jax.numpy as jnp
import equinox as eqx
import chex
from .convolutional import build_convnet
from dataclasses import dataclass
from src.custom_types import *

@dataclass
class SelfAttentionBlockConfig:
    embed_dim: int
    num_heads: int
    mlp_ratio: int = 4

@dataclass
class DecoderConfig:
    output_size: int
    embed_dim: int
    num_heads: int
    num_blocks: int
    mlp_ratio: int = 4
    final_mlp_depth: int=1

class SelfAttentionBlock(eqx.Module):
    self_attn: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(self, config: SelfAttentionBlockConfig, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)
        
        self.self_attn = eqx.nn.MultiheadAttention(
            num_heads=config.num_heads, query_size=config.embed_dim, key=k1
        )
        
        mlp_hidden = config.embed_dim * config.mlp_ratio
        self.mlp = eqx.nn.MLP(
            in_size=config.embed_dim, 
            out_size=config.embed_dim, 
            width_size=mlp_hidden, 
            depth=1, 
            activation=jax.nn.gelu, 
            key=k2
        )
        self.norm1 = eqx.nn.LayerNorm(config.embed_dim)
        self.norm2 = eqx.nn.LayerNorm(config.embed_dim)

    def __call__(self, x: chex.Array) -> chex.Array:
        x_norm1 = jax.vmap(self.norm1)(x)
        attn_out = self.self_attn(query=x_norm1, key_=x_norm1, value=x_norm1)
        x = x + attn_out

        x_norm2 = jax.vmap(self.norm2)(x)
        mlp_out = jax.vmap(self.mlp)(x_norm2)

        return x + mlp_out


class ObservationPolicyNetwork(eqx.Module):
    cnn: eqx.Module
    proj: eqx.nn.Conv2d
    norm: eqx.nn.LayerNorm
    norm_final: eqx.nn.LayerNorm
    blocks: tuple[SelfAttentionBlock, ...]
    action_head: eqx.Module

    def __init__(self, cnn_config_name: str, decoder_config_name: str, seed: int = 777):
        decoder_config = decoder_config_register[decoder_config_name]
        
        keys = jax.random.split(jax.random.key(seed), decoder_config.num_blocks + 3)
        self.cnn = build_convnet(keys[0], config_name=cnn_config_name)

        cnn_out_channels = self.cnn.out_channels

        self.proj = eqx.nn.Conv2d(
            in_channels=cnn_out_channels + 2, 
            out_channels=decoder_config.embed_dim, 
            kernel_size=1, 
            key=keys[1]
        )

        self.norm = eqx.nn.LayerNorm(decoder_config.embed_dim)

        block_config = SelfAttentionBlockConfig(
            embed_dim=decoder_config.embed_dim, 
            num_heads=decoder_config.num_heads, 
            mlp_ratio=decoder_config.mlp_ratio
        )
        
        self.blocks = tuple([
            SelfAttentionBlock(block_config, key=keys[i+2]) 
            for i in range(decoder_config.num_blocks)
        ])
        
        self.norm_final = eqx.nn.LayerNorm(decoder_config.embed_dim)
        
        # Action head predicts the continuous action from the final pos representation
        self.action_head = eqx.nn.MLP(
            in_size=decoder_config.embed_dim, 
            out_size=3, 
            width_size=decoder_config.embed_dim, 
            depth=decoder_config.final_mlp_depth, 
            activation=jax.nn.gelu, 
            key=keys[-1]
        )

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

        # x here is the decoder input
        x = jax.vmap(self.norm)(flattened.T)        
        
        for block in self.blocks:
            x = block(x)

        action = self.action_head(self.norm_final(x[0]))
        return action
    
class ObservationPolicy(eqx.Module):
    '''
    Oracle policies also has access to underlying environment state on top of its own state and the observation.
    Agent policies should never use env_state !
    '''
    observation_policy_net: ObservationPolicyNetwork

    def __init__(self, agent_policy_net) -> None:
        super().__init__()
        self.observation_policy_net = agent_policy_net

    def __call__(self, rng_key: Key, policy_state: PolicyState, env_state: EnvState, observation: FullCanvas, env_params: EnvParams) -> Tuple[PolicyState,Action]:
        return policy_state, self.observation_policy_net(observation)


decoder_config_register = {
    "small": DecoderConfig(output_size=3, embed_dim=64, num_heads=4, num_blocks=2, final_mlp_depth=1),
    "medium": DecoderConfig(output_size=3, embed_dim=128, num_heads=4, num_blocks=4, final_mlp_depth=1),
    "big": DecoderConfig(output_size=3, embed_dim=256, num_heads=8, num_blocks=6, final_mlp_depth=1),
    "no_attention": DecoderConfig(output_size=3, embed_dim=256, num_heads=1, num_blocks=0, final_mlp_depth=4),
    "tiny": DecoderConfig(output_size=3, embed_dim=64, num_heads=1, num_blocks=0, final_mlp_depth=1),
    "single_layer_mlp": DecoderConfig(output_size=3, embed_dim=256, num_heads=1, num_blocks=0, final_mlp_depth=1),
}
