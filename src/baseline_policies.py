import jax
import jax.numpy as jnp
from .custom_types import *

def random_agent_policy(rng_key: Key, agent_state: PolicyState, env_state: EnvState, observation: FullCanvas, env_params: EnvParams) -> Tuple[PolicyState, Action]:
    stroke_max_length = env_params.stroke_max_length
    movement_key, pressure_key = jax.random.split(rng_key)
    movement = jax.random.uniform(movement_key, shape=(2,), minval=-stroke_max_length, maxval=stroke_max_length, dtype=jnp.float32)

    # NOTE: pressure threshold is set 0. by convention
    pressure = jax.random.bernoulli(pressure_key, p=0.5, shape=(1,)) - .5

    return agent_state, jnp.concat((movement, pressure))

def closest_line_policy(rng_key: Key, policy_state: PolicyState, env_state: EnvState, observation: FullCanvas, env_params: EnvParams, precision: float=1e-4) -> Tuple[PolicyState, Action]:
    """
    Precision parameter controls how close we need to get to endpoint before starting the stroke
    Set it low enough that it forces to "finetune" the starting position (avoids low total rewards due to ambguity)
    """
    # Make a (2S, 2) list of strokes
    endpoints: Float[Array, "2S 2"] = jnp.concat(jnp.split(env_state.target_strokes, 2, -1), axis=0)
    status = jnp.tile(env_state.target_strokes_status, 2)
    line_ids = jnp.tile(jnp.arange(env_params.num_target_strokes), 2)

    dists = jnp.max(jnp.abs(env_state.position[None, :] - endpoints), -1)

    filtered_l2_dists = jnp.where(status, 2**10, dists) 
    is_close_to_endpoint = (jnp.min(filtered_l2_dists) < precision)
    target_line_id = line_ids[jnp.argmin(filtered_l2_dists)]
    point1, point2 = jnp.split(env_state.target_strokes[target_line_id], 2)
    dist1, dist2 = jnp.max(jnp.abs(env_state.position-point1)), jnp.max(jnp.abs(env_state.position-point2))
    close_point = jnp.where(dist1<=dist2, point1, point2)
    far_point = jnp.where(dist1<=dist2, point2, point1)

    target_endpoint = jnp.where(is_close_to_endpoint, far_point, close_point)

    # Don't move if all lines already done
    movement = jnp.where(jnp.all(status), jnp.zeros(2, dtype=jnp.float32), target_endpoint - env_state.position)


    return policy_state, jnp.concat((movement, jnp.array([is_close_to_endpoint - .5])))
def ordered_lines_policy(rng_key: Key, policy_state: PolicyState, env_state: EnvState, observation: FullCanvas, env_params: EnvParams, precision: float=1e-6) -> Tuple[PolicyState, Action]:
    # This policy assumes lines in env_state are already sorted according to the rule
    all_lines_done = jnp.all(env_state.target_strokes_status)
    next_line_idx = jnp.argmax(~env_state.target_strokes_status) # returns first not done
    next_line_endpoints = env_state.target_strokes[next_line_idx]
    s, e = jnp.split(next_line_endpoints, 2)

    ds_l2 = jnp.sum((env_state.position - s)**2)
    de_l2 = jnp.sum((env_state.position - e)**2)
    is_close_to_s = (ds_l2 < precision) 
    is_close_to_e = (de_l2 < precision) 
    is_closer_to_e = (de_l2 < ds_l2)

    # If close to one, choose the other as target; if not, go to l2_closest
    is_drawing = is_close_to_e | is_close_to_s
    target_drawing =  (is_close_to_s * e) + (is_close_to_e * s)
    target_moving = is_closer_to_e * e + (1-is_closer_to_e) * s 
    target = is_drawing * target_drawing + (1-is_drawing) * target_moving

    movement = jnp.where(all_lines_done, jnp.zeros(2, dtype=jnp.float32), target - env_state.position)
    pressure = jnp.where(all_lines_done, -.5, is_drawing - .5)
    
    return policy_state, jnp.concat((movement, jnp.array([pressure])))

def ordered_lines__ordered_endpoints_policy(rng_key: Key, policy_state: PolicyState, env_state: EnvState, observation: FullCanvas, env_params: EnvParams) -> Tuple[PolicyState, Action]:
    # This policy assumes lines and their endpoints in env_state are already sorted according to the rule
    raise NotImplementedError

def make_custom_noise_level_policy(policy_fn: Policy, noise_level: float):
    def noisy_policy(rng_key: Key, policy_state: PolicyState, env_state: EnvState, observation: FullCanvas, env_params: EnvParams) -> Tuple[PolicyState, Action]:
        _, oracle_actions = policy_fn(rng_key, policy_state, env_state, observation, env_params)
        action_noise = jnp.array([noise_level, noise_level, 0.]) * jax.random.uniform(rng_key, shape=(3,), minval=-1, maxval=1)
        return policy_state, oracle_actions + action_noise 
    return noisy_policy

baseline_policy_register = {
    "random": random_agent_policy,
    "closest": closest_line_policy,
    "ordered": ordered_lines_policy,
}