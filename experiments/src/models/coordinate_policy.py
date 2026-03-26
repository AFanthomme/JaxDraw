import jax
import jax.numpy as jnp
import equinox as eqx
import chex
from dataclasses import dataclass
from typing import Optional

@dataclass
class AttentionBlockConfig:
    embed_dim: int
    num_heads: int
    mlp_ratio: int = 4

@dataclass
class CoordinateBasedPolicyConfig:
    embed_dim: int
    num_heads: int
    num_blocks: int
    action_dim: int = 3
    mlp_ratio: int = 4

class SelfAttentionBlock(eqx.Module):
    self_attn: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(self, config: AttentionBlockConfig, key: jax.Array):
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


class CoordinateBasedPolicy(eqx.Module):
    line_proj: eqx.nn.Linear
    pos_proj: eqx.nn.Linear
    blocks: tuple[SelfAttentionBlock, ...]
    norm_final: eqx.nn.LayerNorm
    action_head: eqx.nn.MLP

    def __init__(self, config: CoordinateBasedPolicyConfig, key: jax.Array):
        keys = jax.random.split(key, config.num_blocks + 3)
        
        # Project lines (x1,y1,x2,y2,line_done) and 2D pos into the same embedding dimension
        # Uing separate embeddings also allows "input type encoding"
        self.line_proj = eqx.nn.Linear(5, config.embed_dim, key=keys[0])
        self.pos_proj = eqx.nn.Linear(2, config.embed_dim, key=keys[1])
        
        block_config = AttentionBlockConfig(
            embed_dim=config.embed_dim, 
            num_heads=config.num_heads, 
            mlp_ratio=config.mlp_ratio
        )
        
        self.blocks = tuple([
            SelfAttentionBlock(block_config, key=keys[i+2]) 
            for i in range(config.num_blocks)
        ])
        
        self.norm_final = eqx.nn.LayerNorm(config.embed_dim)
        
        # Action head predicts the continuous action from the final pos representation
        self.action_head = eqx.nn.MLP(
            in_size=config.embed_dim, 
            out_size=3, 
            width_size=config.embed_dim, 
            depth=1, 
            activation=jax.nn.gelu, 
            key=keys[-1]
        )

    def __call__(self, lines: chex.Array, pos: chex.Array) -> chex.Array:
        line_embeds = jax.vmap(self.line_proj)(lines) 
        pos_embed = self.pos_proj(pos)
        
        # Pos first, same as "CLS", will accumulate information for output
        x = jnp.concatenate([pos_embed[None, :], line_embeds], axis=0) # (6, embed_dim)
        
        # No positional encoding, ensuring permutation invariance for the lines.
        # Also invariant for pos token, but specialization achieved via embedding
        for block in self.blocks:
            x = block(x)
            
        # Use only info accumulated in pos token to compute action
        action = self.action_head(self.norm_final(x[0]))
        return action

policy_config_register = {
    "small": CoordinateBasedPolicyConfig(embed_dim=64, num_heads=4, num_blocks=2),
    "medium": CoordinateBasedPolicyConfig(embed_dim=128, num_heads=4, num_blocks=4),
    "big": CoordinateBasedPolicyConfig(embed_dim=256, num_heads=8, num_blocks=6)
}

def build_coordinate_based_policy(config: Optional[CoordinateBasedPolicyConfig] = None, config_name: Optional[str] = None, seed: int = 777) -> CoordinateBasedPolicy:
    if config is None:
        assert config_name is not None
        config = policy_config_register[config_name]
        
    key = jax.random.PRNGKey(seed)
    return CoordinateBasedPolicy(config, key)