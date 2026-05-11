import jax
import jax.numpy as jnp
import equinox as eqx
import chex
from dataclasses import dataclass
from src.custom_types import *
from src.models.attention_blocks import *

@dataclass
class EnvStateRuleBasedPolicyConfig:
    embed_dim: int
    num_heads: int
    num_blocks_full_attn: int
    num_blocks_cross_attn: int
    rule_encoder_depth: int
    action_dim: int = 3
    mlp_ratio: int = 4

class EnvStateRuleBasedPolicyNetwork(eqx.Module):
    stroke_proj: eqx.nn.Linear
    pos_proj: eqx.nn.Linear
    self_attn_blocks: tuple[SelfAttentionBlock, ...]
    cross_attn_blocks: tuple[CrossAttentionBlock, ...]
    norm_final: eqx.nn.LayerNorm
    action_head: eqx.nn.MLP
    rule_encoder: eqx.nn.MLP

    def __init__(self, config: EnvStateRuleBasedPolicyConfig, key: jax.Array):
        keys = jax.random.split(key, config.num_blocks_full_attn + config.num_blocks_cross_attn + 4)
        
        # These are for geometric Self-Attention part
        self.stroke_proj = eqx.nn.Linear(5, config.embed_dim, key=keys[0])
        self.pos_proj = eqx.nn.Linear(2, config.embed_dim, key=keys[1])
        self.rule_encoder = eqx.nn.MLP(
            in_size=10, # 5 modes as 1-hot, ref_point and ref_angle as 2d vectors, decreasing flag 
            out_size=config.embed_dim, 
            width_size=config.embed_dim, 
            depth=config.rule_encoder_depth, 
            activation=jax.nn.gelu, 
            key=keys[2]
        )
        self.action_head = eqx.nn.MLP(
            in_size=config.embed_dim, 
            out_size=3, 
            width_size=config.embed_dim, 
            depth=1, 
            activation=jax.nn.gelu, 
            key=keys[3]
        )

        block_config = AttentionBlockConfig(
            embed_dim=config.embed_dim, 
            num_heads=config.num_heads, 
            mlp_ratio=config.mlp_ratio
        )
        
        self.self_attn_blocks = tuple([
            SelfAttentionBlock(block_config, key=keys[i+4]) 
            for i in range(config.num_blocks_full_attn)
        ])

        self.cross_attn_blocks = tuple([
            CrossAttentionBlock(block_config, key=keys[i+config.num_blocks_full_attn+4]) 
            for i in range(config.num_blocks_cross_attn)
        ])

        self.norm_final = eqx.nn.LayerNorm(config.embed_dim)
        

    def __call__(self, env_state: EnvState) -> chex.Array:
        # First, the geometric self-attention part
        strokes = env_state.target_strokes
        strokes_done = env_state.target_strokes_status
        full_stroke_infos = jnp.concatenate([strokes, strokes_done[:, None]], axis=-1)
        stroke_embeds = jax.vmap(self.stroke_proj)(full_stroke_infos) # n_lines, embed_dim
        position_embed = self.pos_proj(env_state.position)

        # Position kept as a separate token, put at the beginning of the sequence
        x = jnp.concatenate([position_embed[None, :], stroke_embeds], axis=0) # (n_lines+1, embed_dim)
        
        # No positional encoding, ensuring permutation invariance for the lines.
        # Also invariant for pos token, but specialization achieved via embedding
        for self_attn_block in self.self_attn_blocks:
            x = self_attn_block(x)

        # Then, cross-attention with the rule
        angle_embed = jnp.array([jnp.cos(env_state.ref_angle), jnp.sin(env_state.ref_angle)])
        mode_embed = jax.nn.one_hot(env_state.sort_mode, 5)
        rule_embed = self.rule_encoder(jnp.concatenate([mode_embed, env_state.decreasing[None], angle_embed, env_state.ref_point]))[None, :]
        
        # Only rule token is updated by retrieving info from geometric features 
        for cross_attn_block in self.cross_attn_blocks:
            rule_embed = cross_attn_block(rule_embed, x)

        action = self.action_head(self.norm_final(rule_embed[0]))
        return action


class EnvStateRuleBasedPolicy(eqx.Module):
    envstate_policy_net: EnvStateRuleBasedPolicyNetwork

    def __init__(self, agent_policy_net) -> None:
        super().__init__()
        self.envstate_policy_net = agent_policy_net

    def __call__(self, rng_key: Key, policy_state: PolicyState, env_state: EnvState, observation: FullCanvas, env_params: EnvParams) -> Tuple[PolicyState,Action]:
        return policy_state, self.envstate_policy_net(env_state)
    
envstate_rule_based_policy_config_register = {
    "small": EnvStateRuleBasedPolicyConfig(embed_dim=256, num_heads=8, num_blocks_full_attn=2, num_blocks_cross_attn=1, rule_encoder_depth=1),
    "small_wide": EnvStateRuleBasedPolicyConfig(embed_dim=512, num_heads=8, num_blocks_full_attn=2, num_blocks_cross_attn=1, rule_encoder_depth=1),
    "medium": EnvStateRuleBasedPolicyConfig(embed_dim=256, num_heads=8, num_blocks_full_attn=3, num_blocks_cross_attn=2, rule_encoder_depth=2),
    "big": EnvStateRuleBasedPolicyConfig(embed_dim=512, num_heads=16, num_blocks_full_attn=4, num_blocks_cross_attn=3, rule_encoder_depth=3),
}
