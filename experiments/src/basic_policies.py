import jax
import jax.numpy as jnp
from .custom_types import *

def random_agent_policy(rng_key: Key, agent_state: PolicyState, env_state: EnvState, observation: FullCanvas, env_params: EnvParams) -> Tuple[PolicyState, Action]:
    """
    Here for testing / interface clarification only.
    """
    stroke_max_length = env_params.stroke_max_length
    movement_key, pressure_key = jax.random.split(rng_key)
    movement = jax.random.uniform(movement_key, shape=(2,), minval=-stroke_max_length, maxval=stroke_max_length, dtype=jnp.float32)

    # NOTE: pressure threshold assumed to be at 0 for easier interfacing with tanh output
    pressure = jax.random.bernoulli(pressure_key, p=0.5, shape=(1,)) - .5

    # NOTE: This is where policy would get updated
    new_policy_state = agent_state
    return new_policy_state, jnp.concat((movement, pressure))

def oracle_policy(rng_key: Key, policy_state: PolicyState, env_state: EnvState, observation: FullCanvas, env_params: EnvParams) -> Tuple[PolicyState, Action]:
    """
    Having access to env_state is cheating, but that's what Oracles are all about
    """
    # Make a (2S, 2) list of strokes
    endpoints = jnp.concat(jnp.split(env_state.target_strokes, 2, -1), axis=0)
    status = jnp.tile(env_state.target_strokes_status, 2)
    line_ids = jnp.tile(jnp.arange(env_params.num_target_strokes), 2)
    position: Coordinate = env_state.position

    # Presure is a float, pos for pen_down neg otherwise; pen_down is a bool
    # Switch to L_infty norm for easier perturbation here
    l_infty_dists = jnp.max(jnp.abs(position[None, :] - endpoints), -1)
    filtered_l_infty_dists = jnp.where(status, 2**10, l_infty_dists) 
    pressure = (jnp.min(filtered_l_infty_dists) < env_params.quality_max_pos_dif) - .5
    pen_is_down = pressure > 0

    # Keep l2 for choosing target points when moving to distant line (ie not drawing)
    l2_dists = jnp.max(jnp.abs(position[None, :] - endpoints), -1)
    # This is to ensure no funny business between l_infty and l2
    # If we're in close infty vicinity of an endpoint, then use that as metric 
    # (and the other endpoint will necessarily be far in that norm because of min_stroke_length)
    dists = jnp.where(pen_is_down, l_infty_dists, l2_dists) 
    filtered_l2_dists = jnp.where(status, 2**10, dists) 
    target_line_id = line_ids[jnp.argmin(filtered_l2_dists)]
    point1, point2 = jnp.split(env_state.target_strokes[target_line_id], 2)
    dist1, dist2 = jnp.max(jnp.abs(position-point1)), jnp.max(jnp.abs(position-point2))
    close_point = jnp.where(dist1<=dist2, point1, point2)
    far_point = jnp.where(dist1<=dist2, point2, point1)

    target_endpoint = jnp.where(pen_is_down, far_point, close_point)

    # Don't move if all lines already done
    movement = jnp.where(jnp.all(status), jnp.zeros(2, dtype=jnp.float32), target_endpoint - position)

    return policy_state, jnp.concat((movement, jnp.array([pressure])))

def noisy_oracle_policy(rng_key: Key, policy_state: PolicyState, env_state: EnvState, observation: FullCanvas, env_params: EnvParams) -> Tuple[PolicyState, Action]:
    """
    Allows for slightly imperfect trajectories
    NOTE: this can lead to widely diverging trajectories if final endpoint of one line close two other endpoints !
    """
    _, oracle_actions = oracle_policy(rng_key, policy_state, env_state, observation, env_params)
    action_noise = jnp.array([0.9, 0.9, 0.]) * env_params.quality_max_pos_dif * jax.random.uniform(rng_key, shape=(3,), minval=-1, maxval=1)
    return policy_state, oracle_actions + action_noise

