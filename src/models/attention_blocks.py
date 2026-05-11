import jax
import equinox as eqx
import chex
from dataclasses import dataclass

@dataclass
class AttentionBlockConfig:
    embed_dim: int
    num_heads: int
    mlp_ratio: int = 4

@dataclass
class EnvStateBasedPolicyConfig:
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

class CrossAttentionBlock(eqx.Module):
    cross_attn: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP
    norm_q: eqx.nn.LayerNorm
    norm_kv: eqx.nn.LayerNorm
    norm_mlp: eqx.nn.LayerNorm

    def __init__(self, config: AttentionBlockConfig, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)
        
        self.cross_attn = eqx.nn.MultiheadAttention(
            num_heads=config.num_heads, 
            query_size=config.embed_dim, 
            key=k1
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
        
        self.norm_q = eqx.nn.LayerNorm(config.embed_dim)
        self.norm_kv = eqx.nn.LayerNorm(config.embed_dim)
        self.norm_mlp = eqx.nn.LayerNorm(config.embed_dim)

    def __call__(self, queries: chex.Array, features: chex.Array) -> chex.Array:
        q_norm = jax.vmap(self.norm_q)(queries)
        kv_norm = jax.vmap(self.norm_kv)(features)
        attn_out = self.cross_attn(query=q_norm, key_=kv_norm, value=kv_norm)
        x = queries + attn_out
        x_norm_mlp = jax.vmap(self.norm_mlp)(x)
        mlp_out = jax.vmap(self.mlp)(x_norm_mlp)
        return x + mlp_out