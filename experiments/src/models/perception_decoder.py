import jax
import jax.numpy as jnp
import equinox as eqx
import chex
from dataclasses import dataclass
from typing import Optional
from jaxtyping import Key

@dataclass(frozen=True)
class PerceptionDecoderConfig:
    query_groups: tuple[tuple[int, int], ...]
    embed_dim: int = 256
    num_heads: int = 8
    num_blocks: int = 4
    mlp_ratio: int = 4

class DETRDecoderBlock(eqx.Module):
    self_attn: eqx.nn.MultiheadAttention
    cross_attn: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    norm3: eqx.nn.LayerNorm
    norm4: eqx.nn.LayerNorm

    def __init__(self, config: PerceptionDecoderConfig, key: Key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.self_attn = eqx.nn.MultiheadAttention(
            num_heads=config.num_heads, query_size=config.embed_dim, key=k1
        )
        self.cross_attn = eqx.nn.MultiheadAttention(
            num_heads=config.num_heads, query_size=config.embed_dim, key=k2
        )
        self.mlp = eqx.nn.MLP(
            in_size=config.embed_dim, out_size=config.embed_dim, 
            width_size=config.embed_dim * config.mlp_ratio, depth=1, 
            activation=jax.nn.gelu, key=k3
        )
        self.norm1 = eqx.nn.LayerNorm(config.embed_dim)
        self.norm2 = eqx.nn.LayerNorm(config.embed_dim)
        self.norm3 = eqx.nn.LayerNorm(config.embed_dim)
        self.norm4 = eqx.nn.LayerNorm(config.embed_dim)

    def __call__(self, queries: chex.Array, memory: chex.Array) -> chex.Array:
        # Self-Attention (Queries)
        q_norm1 = jax.vmap(self.norm1)(queries)
        queries = queries + self.self_attn(query=q_norm1, key_=q_norm1, value=q_norm1)
        # Cross-Attention 
        q_norm2 = jax.vmap(self.norm2)(queries)
        m_norm = jax.vmap(self.norm3)(memory) 
        queries = queries + self.cross_attn(query=q_norm2, key_=m_norm, value=m_norm)
        # MLP
        q_norm3 = jax.vmap(self.norm4)(queries)
        queries = queries + jax.vmap(self.mlp)(q_norm3)
        return queries

class PerceptionDecoder(eqx.Module):
    # Allow for more flexible grouping of queries / outputs
    output_heads: tuple[eqx.nn.Linear, ...]
    embed_dim: int
    query_embed: eqx.nn.Embedding
    blocks: tuple[DETRDecoderBlock, ...]
    norm_final: eqx.nn.LayerNorm
    # Static metadata
    total_queries: int
    split_indices: tuple[int, ...]

    def __init__(self, config: PerceptionDecoderConfig, key: jax.Array):
        keys = jax.random.split(key, config.num_blocks + len(config.query_groups) + 1)
        self.embed_dim = config.embed_dim
        
        self.total_queries = sum(n for n, _ in config.query_groups)
        
        splits = [n for n, _ in config.query_groups][:-1]
        cum_splits = []
        current = 0
        for s in splits:
            current += s
            cum_splits.append(current)
        self.split_indices = tuple(cum_splits)

        self.query_embed = eqx.nn.Embedding(self.total_queries, config.embed_dim, key=keys[0])
        self.blocks = tuple(DETRDecoderBlock(config, key=keys[i+1]) for i in range(config.num_blocks))
        self.norm_final = eqx.nn.LayerNorm(config.embed_dim)
        
        # Build a separate linear head for each query group
        head_keys = keys[config.num_blocks + 1:]
        heads = []
        for i, (_, out_dim) in enumerate(config.query_groups):
            heads.append(
                eqx.nn.MLP(
                in_size=config.embed_dim, 
                out_size=out_dim, 
                width_size=config.embed_dim, 
                depth=1, 
                activation=jax.nn.gelu, 
                key=head_keys[i]
            ))
        self.output_heads = tuple(heads)

    def __call__(self, memory: chex.Array) -> tuple[chex.Array, ...]:
        queries = jax.vmap(self.query_embed)(jnp.arange(self.total_queries))
        
        for block in self.blocks:
            queries = block(queries, memory)
            
        queries = jax.vmap(self.norm_final)(queries)
        
        # Split the queries back into their groups (jnp.split requires cumulative indices)
        grouped_queries = jnp.split(queries, self.split_indices, axis=0)
        
        outputs = []
        for head, group in zip(self.output_heads, grouped_queries):
            outputs.append(jax.vmap(head)(group))
            
        return tuple(outputs)
    
def build_decoder(query_groups: tuple[tuple[int, int], ...], key: Key):
    config = PerceptionDecoderConfig(query_groups)
    return PerceptionDecoder(config, key)