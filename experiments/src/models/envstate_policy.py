import jax
import jax.numpy as jnp
import equinox as eqx
import chex
from dataclasses import dataclass
from typing import Optional
from src.custom_types import *

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


class EnvStateBasedPolicyNetwork(eqx.Module):
    line_proj: eqx.nn.Linear
    pos_proj: eqx.nn.Linear
    blocks: tuple[SelfAttentionBlock, ...]
    norm_final: eqx.nn.LayerNorm
    action_head: eqx.nn.MLP
    rule_angle_proj: eqx.nn.Linear
    rule_point_proj: eqx.nn.Linear
    rule_mode_proj: eqx.nn.Embedding

    def __init__(self, config: EnvStateBasedPolicyConfig, key: jax.Array):
        keys = jax.random.split(key, config.num_blocks + 6)
        
        # Project lines (x1,y1,x2,y2,line_done) and 2D pos into the same embedding dimension
        # Uing separate embeddings also allows "input type encoding"
        self.line_proj = eqx.nn.Linear(5, config.embed_dim, key=keys[0])
        self.pos_proj = eqx.nn.Linear(2, config.embed_dim, key=keys[1])
        self.rule_angle_proj = eqx.nn.Linear(2, config.embed_dim, key=keys[2])
        self.rule_point_proj = eqx.nn.Linear(2, config.embed_dim, key=keys[3])
        self.rule_mode_proj = eqx.nn.Embedding(5, config.embed_dim, key=keys[4])
        
        block_config = AttentionBlockConfig(
            embed_dim=config.embed_dim, 
            num_heads=config.num_heads, 
            mlp_ratio=config.mlp_ratio
        )
        
        self.blocks = tuple([
            SelfAttentionBlock(block_config, key=keys[i+5]) 
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

    def __call__(self, lines, pos, sort_mode, ref_angle, ref_point) -> chex.Array:
        line_embeds = jax.vmap(self.line_proj)(lines) 
        pos_embed = self.pos_proj(pos)
        angle_embed = self.rule_angle_proj(jnp.array([jnp.cos(ref_angle), jnp.sin(ref_angle)]))
        point_embed = self.rule_point_proj(ref_point)
        mode_embed = self.rule_mode_proj(sort_mode)
        
        # Pos first, same as "CLS", will accumulate information for output
        x = jnp.concatenate([pos_embed[None, :], angle_embed[None, :], point_embed[None, :], mode_embed[None, :], line_embeds], axis=0) # (6, embed_dim)
        
        # No positional encoding, ensuring permutation invariance for the lines.
        # Also invariant for pos token, but specialization achieved via embedding
        for block in self.blocks:
            x = block(x)
            
        # Use only info accumulated in pos token to compute action
        action = self.action_head(self.norm_final(x[0]))
        return action

policy_config_register = {
    "small": EnvStateBasedPolicyConfig(embed_dim=128, num_heads=4, num_blocks=2),
    "small_ish": EnvStateBasedPolicyConfig(embed_dim=256, num_heads=4, num_blocks=3),
    "medium": EnvStateBasedPolicyConfig(embed_dim=256, num_heads=4, num_blocks=4),
    "big": EnvStateBasedPolicyConfig(embed_dim=256, num_heads=8, num_blocks=6)
}

def build_envstate_policy(config: Optional[EnvStateBasedPolicyConfig] = None, config_name: Optional[str] = None, seed: int = 777) -> EnvStateBasedPolicyNetwork:
    if config is None:
        assert config_name is not None
        config = policy_config_register[config_name]
        
    key = jax.random.PRNGKey(seed)
    return EnvStateBasedPolicyNetwork(config, key)

class EnvStatePolicy(eqx.Module):
    '''
    Oracle policies also has access to underlying environment state on top of its own state and the observation.
    Agent policies should never use env_state !
    '''
    envstate_policy_net: EnvStateBasedPolicyNetwork

    def __init__(self, agent_policy_net) -> None:
        super().__init__()
        self.envstate_policy_net = agent_policy_net

    def __call__(self, rng_key: Key, policy_state: PolicyState, env_state: EnvState, observation: FullCanvas, env_params: EnvParams) -> Tuple[PolicyState,Action]:
        strokes = env_state.target_strokes
        statuses = env_state.target_strokes_status
        pos = env_state.position
        sort_mode = env_state.sort_mode
        ref_angle = env_state.ref_angle
        ref_point = env_state.ref_point
        lines = jnp.concatenate([strokes, statuses[:,None]], axis=-1)
         
        return policy_state, self.envstate_policy_net(lines, pos, sort_mode, ref_angle, ref_point)
