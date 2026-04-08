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

def closest_line_policy(rng_key: Key, policy_state: PolicyState, env_state: EnvState, observation: FullCanvas, env_params: EnvParams) -> Tuple[PolicyState, Action]:
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
    pressure = (jnp.min(filtered_l_infty_dists) < env_params.line_done_cutoff) - .5
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

def ordered_lines_policy(rng_key: Key, policy_state: PolicyState, env_state: EnvState, observation: FullCanvas, env_params: EnvParams) -> Tuple[PolicyState, Action]:
    # This policy assumes lines in env_state are already sorted according to the rule
    all_lines_done = jnp.all(env_state.target_strokes_status)
    next_line_idx = jnp.argmax(~env_state.target_strokes_status) # returns first not done
    next_line_endpoints = env_state.target_strokes[next_line_idx]
    s, e = jnp.split(next_line_endpoints, 2)

    ds_l2 = jnp.sum((env_state.position - s)**2)
    ds_linf = jnp.max(jnp.abs(env_state.position - s))
    de_l2 = jnp.sum((env_state.position - e)**2)
    de_linf = jnp.max(jnp.abs(env_state.position - e))
    is_linf_close_to_s = (ds_linf < env_params.line_done_cutoff) 
    is_linf_close_to_e = (de_linf < env_params.line_done_cutoff) 
    is_l2_closer_to_e = (de_l2 < ds_l2)

    # If close to one, choose the other as target; if not, go to l2_closest
    is_drawing = is_linf_close_to_e | is_linf_close_to_s
    target_drawing =  (is_linf_close_to_s * e) + (is_linf_close_to_e * s)
    target_moving = is_l2_closer_to_e * e + (1-is_l2_closer_to_e) * s 
    target = is_drawing * target_drawing + (1-is_drawing) * target_moving

    movement = jnp.where(all_lines_done, jnp.zeros(2, dtype=jnp.float32), target - env_state.position)
    pressure = jnp.where(all_lines_done, 0., is_drawing - .5)
    
    return policy_state, jnp.concat((movement, jnp.array([pressure])))

def ordered_lines__ordered_endpoints_policy(rng_key: Key, policy_state: PolicyState, env_state: EnvState, observation: FullCanvas, env_params: EnvParams) -> Tuple[PolicyState, Action]:
    # This policy assumes lines and their endpoints in env_state are already sorted according to the rule
    raise NotImplementedError

def make_custom_noise_level_policy(policy_fn: Policy, noise_level: float):
    def f(rng_key: Key, policy_state: PolicyState, env_state: EnvState, observation: FullCanvas, env_params: EnvParams) -> Tuple[PolicyState, Action]:
        """
        Allows for slightly imperfect trajectories
        NOTE: this can lead to widely diverging trajectories if final endpoint of one line close two other endpoints !
        """
        _, oracle_actions = policy_fn(rng_key, policy_state, env_state, observation, env_params)
        action_noise = jnp.array([noise_level, noise_level, 0.]) * jax.random.uniform(rng_key, shape=(3,), minval=-1, maxval=1)
        return policy_state, oracle_actions + action_noise 
    return f

baseline_policy_register = {
         "random": random_agent_policy,
         "closest": closest_line_policy,
         "noisy_closest": make_custom_noise_level_policy(closest_line_policy, 1./128),
         "noisy_closest_001": make_custom_noise_level_policy(closest_line_policy, 0.01), 
         "noisy_closest_002": make_custom_noise_level_policy(closest_line_policy, 0.02), 
         "noisy_closest_003": make_custom_noise_level_policy(closest_line_policy, 0.03), 
         "noisy_closest_006": make_custom_noise_level_policy(closest_line_policy, 0.06), 
         "ordered_lines_policy": ordered_lines_policy,
         "noisy_ordered": make_custom_noise_level_policy(ordered_lines_policy, 1./128),
}