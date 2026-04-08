import jax
import jax.numpy as jnp
from typing import cast
from .custom_types import *
import numpy as np

def _compute_keys(
    strokes: jnp.ndarray, 
    sort_mode: jnp.ndarray, 
    theta: jnp.ndarray, 
    ref_point: jnp.ndarray,
    is_decreasing: jnp.ndarray
) -> jnp.ndarray:
    
    x1, y1, x2, y2 = strokes[..., 0], strokes[..., 1], strokes[..., 2], strokes[..., 3]
    
    # Precompute shared geometric properties to avoid redundant compute in branches
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dx = x2 - x1
    dy = y2 - y1
    
    # Format ref_point for broadcasting (..., N)
    # Using jnp.atleast_1d and expand_dims ensures safety across batch ranks
    x_ref = jnp.expand_dims(ref_point[..., 0], axis=-1)
    y_ref = jnp.expand_dims(ref_point[..., 1], axis=-1)

    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)

    def mode_0_projection():
        return cx * cos_t + cy * sin_t

    def mode_1_proximity_ref():
        return (cx - x_ref)**2 + (cy - y_ref)**2

    def mode_2_stroke_length():
        return dx**2 + dy**2

    def mode_3_spiral_ego(): 
        # Egocentric orientation
        sim = ((dx * cos_t + dy * sin_t)**2) / (dx**2 + dy**2 + 1e-7)
        return -sim 
    
    def mode_4_spiral_allo():
        # Allocentric polar angle around frame center (0.5, 0.5)
        # jnp.arctan2 yields [-pi, pi].
        angle = jnp.arctan2(cy - 0.5, cx - 0.5)
        # Difference from theta, wrapped to [0, 2pi) to create a clean radial sweep
        return (angle - theta) % (2 * jnp.pi)

    branches = (
            mode_0_projection,
            mode_1_proximity_ref,
            mode_2_stroke_length,
            mode_3_spiral_ego,
            mode_4_spiral_allo
        )

    base_keys = jax.lax.switch(sort_mode, branches)
    multiplier = jnp.where(is_decreasing, -1.0, 1.0)
    return base_keys * multiplier

def _reorder_strokes(
    strokes: jnp.ndarray, 
    sort_mode: jnp.ndarray, 
    theta: jnp.ndarray, 
    ref_point: jnp.ndarray,
    is_decreasing: jnp.ndarray
) -> jnp.ndarray:
    keys = _compute_keys(strokes, sort_mode, theta, ref_point, is_decreasing)
    indices = jnp.argsort(keys, axis=-1)
    indices_expanded = jnp.expand_dims(indices, axis=-1)
    sorted_strokes = jnp.take_along_axis(strokes, indices_expanded, axis=-2)
    return sorted_strokes

def _get_compositional_rule_strokes(
    sort_mode: jnp.ndarray, 
    theta: jnp.ndarray, 
    ref_point: jnp.ndarray, 
    is_decreasing: jnp.ndarray
) -> jnp.ndarray:
    c = 0.5            # Canvas center
    L = 0.3           # Main vector length
    Lh = 0.1          # Arrowhead length
    Lm = 0.06          # Modifier bar length
    alpha = jnp.pi / 6 # Arrowhead spread
    
    off = jnp.array([-1.0, -1.0, -1.0, -1.0])
    dec_f = is_decreasing.astype(jnp.float32)

    # Pre-calculate shared angle tokens
    tip_x, tip_y = c + L*jnp.cos(theta), c + L*jnp.sin(theta)
    vec_base = jnp.array([c, c, tip_x, tip_y])
    
    # Arrowheads: Flip inward if decreasing
    dir_sign = jnp.where(is_decreasing, -1.0, 1.0) 
    arr1 = jnp.array([tip_x, tip_y, tip_x - dir_sign*Lh*jnp.cos(theta+alpha), tip_y - dir_sign*Lh*jnp.sin(theta+alpha)])
    arr2 = jnp.array([tip_x, tip_y, tip_x - dir_sign*Lh*jnp.cos(theta-alpha), tip_y - dir_sign*Lh*jnp.sin(theta-alpha)])

    # Perpendicular angle for modifiers
    perp_t = theta + jnp.pi / 2

    # --- Branches ---
    def draw_0_proj(op):
        return jnp.stack([vec_base, arr1, arr2, off, off, off, off, off])

    def draw_1_prox_ref(op):
        rx, ry = ref_point[0], ref_point[1]
        # + becomes x if decreasing
        rot = dec_f * (jnp.pi / 4.0) 
        r1 = jnp.array([rx - Lm*jnp.cos(rot), ry - Lm*jnp.sin(rot), rx + Lm*jnp.cos(rot), ry + Lm*jnp.sin(rot)])
        r2 = jnp.array([rx - Lm*jnp.cos(rot+jnp.pi/2), ry - Lm*jnp.sin(rot+jnp.pi/2), rx + Lm*jnp.cos(rot+jnp.pi/2), ry + Lm*jnp.sin(rot+jnp.pi/2)])
        return jnp.stack([r1, r2, off, off, off, off, off, off])

    def draw_2_length(op):
        # A horizontal line with two vertical bounds (like a ruler)
        # Rotates to vertical if decreasing
        rot = dec_f * (jnp.pi / 2.0)
        l1 = jnp.array([c - L*jnp.cos(rot), c - L*jnp.sin(rot), c + L*jnp.cos(rot), c + L*jnp.sin(rot)])
        b1 = jnp.array([c - L*jnp.cos(rot) - Lh*jnp.cos(rot+jnp.pi/2), c - L*jnp.sin(rot) - Lh*jnp.sin(rot+jnp.pi/2), 
                        c - L*jnp.cos(rot) + Lh*jnp.cos(rot+jnp.pi/2), c - L*jnp.sin(rot) + Lh*jnp.sin(rot+jnp.pi/2)])
        b2 = jnp.array([c + L*jnp.cos(rot) - Lh*jnp.cos(rot+jnp.pi/2), c + L*jnp.sin(rot) - Lh*jnp.sin(rot+jnp.pi/2), 
                        c + L*jnp.cos(rot) + Lh*jnp.cos(rot+jnp.pi/2), c + L*jnp.sin(rot) + Lh*jnp.sin(rot+jnp.pi/2)])
        return jnp.stack([l1, b1, b2, off, off, off, off, off])

    def draw_3_ego(op):
        # Orthogonal crossbar at the vector's midpoint
        mid_x, mid_y = c + (L/2)*jnp.cos(theta), c + (L/2)*jnp.sin(theta)
        ego_mod = jnp.array([mid_x - Lm*jnp.cos(perp_t), mid_y - Lm*jnp.sin(perp_t), mid_x + Lm*jnp.cos(perp_t), mid_y + Lm*jnp.sin(perp_t)])
        return jnp.stack([vec_base, arr1, arr2, ego_mod, off, off, off, off])

    def draw_4_allo(op):
        # Tangent bar at the vector's tip
        allo_mod = jnp.array([tip_x - Lm*jnp.cos(perp_t), tip_y - Lm*jnp.sin(perp_t), tip_x + Lm*jnp.cos(perp_t), tip_y + Lm*jnp.sin(perp_t)])
        return jnp.stack([vec_base, arr1, arr2, allo_mod, off, off, off, off])

    branches = (draw_0_proj, draw_1_prox_ref, draw_2_length, draw_3_ego, draw_4_allo)
    return jax.lax.switch(sort_mode, branches, None)

def _dist_to_segments(frame_coordinates: Float[Array, "H W 2"], strokes: Float[Array, "N 4"]) -> Float[Array, "H W N"]:
    """
    Compute signed distance fields for N strokes at once
    """
    ends, starts = jnp.split(strokes, 2, axis=-1) 
    line_vecs: Float[Array, "N 2"] = ends-starts
    line_square_norms: Float[Array, "N"] = jnp.sum(line_vecs**2, axis=-1) + 1e-6 
    vec_to_start: Float[Array, "H W N 2"] = frame_coordinates[:,:,None,:] - starts[None, None, :, :] 

    # Start by finding the projection on the line, then the segment through clipping
    proj_signed_length: Float[Array, "H W N"] = jnp.sum(vec_to_start * line_vecs[None, None, :, :], axis=-1) / line_square_norms[None, None, :]

    # proj_signed_length needs expanding on the x/y dim at the end only, line_vecs across HW
    projection_on_segment: Float[Array, "H W N 2"] = starts[None, None, :, :] + jnp.clip(proj_signed_length[:, :, :, None], 0, 1) * line_vecs[None, None, :, :]

    dist_to_segments = jnp.linalg.norm(frame_coordinates[:, :, None, :] - projection_on_segment, axis=-1) 
    return dist_to_segments

def _draw_strokes_single_canvas(strokes: Float[Array, "N 4"], env_params: EnvParams) -> SingleFrame:
    frame_coordinates = jnp.indices((env_params.size, env_params.size), dtype=jnp.float32).transpose(1, 2, 0) / env_params.size
    dists = _dist_to_segments(frame_coordinates, strokes) 
    frame_pixels: SingleFrame = jnp.clip(1. - jnp.clip((jnp.min(dists, axis=-1) - env_params.thickness) / (env_params.softness+1e-6), 0.0, .99), 0.0, .99) 
    return frame_pixels

def _draw_point_single_canvas(coordinate_vector: Coordinate, env_params: EnvParams) -> SingleFrame:
    """
    Reducing the segment to a point allows to reuse strokes draw call, so do that
    """
    point_as_stroke_collection = jnp.tile(coordinate_vector, 2)[None, :]
    return _draw_strokes_single_canvas(point_as_stroke_collection, env_params) 
    
def _generate_observation(env_state: EnvState, env_params: EnvParams) -> FullCanvas:
    """
    Combines environment state and UI rule overlay.
    """

    target_canvas = _draw_strokes_single_canvas(env_state.target_strokes, env_params)
    draw_canvas = _draw_strokes_single_canvas(env_state.drawn_strokes, env_params)   
    pos_canvas = _draw_point_single_canvas(env_state.position, env_params)           
    
    rule_strokes = _get_compositional_rule_strokes(
        env_state.sort_mode, 
        env_state.ref_angle, 
        env_state.ref_point, 
        env_state.decreasing
    )
    rule_canvas = _draw_strokes_single_canvas(rule_strokes, env_params)
    
    r = pos_canvas + rule_canvas
    g = draw_canvas + rule_canvas
    b = target_canvas + rule_canvas
    
    return jnp.stack((r, g, b), axis=0)

def _is_valid_match(test_stroke: Stroke, target_stroke: Stroke, env_params: EnvParams) -> Tuple[TargetStrokesStatus, Float[Array, "B S"]]: 
    """
    Checks if test_stroke matches ref_stroke based on endpoint proximity
    Tests all pairings of endpoints in parallel, and ensures not both test endpoints are close to the same target
    """
    cutoff = env_params.line_done_cutoff

    # Compute all 4 pairwise squared distances via broadcasting
    t_pts = test_stroke.reshape(2, 2)
    r_pts = target_stroke.reshape(2, 2)
    diffs = t_pts[:, None, :] - r_pts[None, :, :]
    # Use L_infty distance for easier noise manipulation
    dist = jnp.max(jnp.abs(diffs), axis=-1)
    
    # Check that each test endpoint is within threshold of SOME ref endpoint
    min_dists_sq = jnp.min(dist, axis=1)
    within_cutoff = jnp.all(min_dists_sq < cutoff)
    
    # Check that both test endpoints are not closest to the same ref endpoint
    # Also, l1 is what will be used for reward calculation
    l1_dist = jnp.sum(jnp.abs(diffs), axis=-1)
    closest_ref_indices = jnp.argmin(l1_dist, axis=1)
    distinct_endpoints = closest_ref_indices[0] != closest_ref_indices[1]

    # For reward, use the sum of the two endpoint distances 
    dist_for_reward = jnp.sum(jnp.min(l1_dist, axis=1))
    
    return within_cutoff & distinct_endpoints, dist_for_reward

def _compute_new_stroke_statuses(latest_stroke, env_state: EnvState, env_params: EnvParams) -> Tuple[TargetStrokesStatus, Float[Array, "B S"]]:   
    current_matches, dist_for_reward = jax.vmap(_is_valid_match, in_axes=(None, 0, None))(latest_stroke, env_state.target_strokes, env_params)
    old_status = env_state.target_strokes_status
    new_status = current_matches | old_status
    return new_status, dist_for_reward

def _canonicalize_strokes(strokes: Float[Array, "... 4"]) -> Float[Array, "... 4"]:
    """
    Crucial for line decoding if used !! While stroke ordering is arbitrary (and treated as such by attention)
    this is not the case for the indvidual stroke structure which needs to be fixed (not too brittle thanks to line length)
    Reorders x1,y1,x2,y2 so that (x1, y1) is always the lexicographically smaller point
    Should work on both strokes or collections using split/stack
    """
    x1, y1, x2, y2 = jnp.split(strokes, 4, axis=-1)
    is_ordered = (x1 < x2) | ((x1 == x2) & (y1 <= y2))
    new_x1 = jnp.where(is_ordered, x1, x2)
    new_y1 = jnp.where(is_ordered, y1, y2)
    new_x2 = jnp.where(is_ordered, x2, x1)
    new_y2 = jnp.where(is_ordered, y2, y1)
    
    return jnp.concat([new_x1, new_y1, new_x2, new_y2], axis=-1)

def _compute_stroke_and_pos_from_action(env_state: EnvState, old_position: Coordinate, action: Action, env_params: EnvParams) -> Tuple[Stroke, Coordinate]:
    new_position = jnp.clip(env_state.position + action[:2], env_params.thickness, 1.-env_params.thickness) 
    proposed_line = jnp.concat((old_position, new_position), axis=-1)
    default_drawn_line_state: Float[Array, "4"] = -10*jnp.ones(4, dtype=jnp.float32)
    pen_is_down = action[2] > 0.
    new_line_state = (1-pen_is_down) * default_drawn_line_state + pen_is_down * proposed_line
    return _canonicalize_strokes(new_line_state), new_position

def _compute_reward_from_action(action: Action, env_state: EnvState, env_params: EnvParams) -> Reward:   
    old_position = env_state.position
    old_status = env_state.target_strokes_status
    current_active = jnp.argmax(~old_status)
    new_stroke, _ = _compute_stroke_and_pos_from_action(env_state, old_position, action, env_params)
    new_status, distances = _compute_new_stroke_statuses(new_stroke, env_state, env_params)

    # Rewarded for having drawn the currently active
    dist_for_reward = distances[current_active]
    reward_bool = new_status[current_active] & ~jnp.all(old_status)
    # Penalized for incorrect drawing (same if ordering error or just invalid line)
    penalty = (action[2] > 0.) * env_params.false_draw_penalty * (reward_bool < .5)
    # Introduce some weighting within the binary reward; decays from 1 to .8 on the valid range, then goes to 0 
    # (avoids harsh penalty for small mistakes, while keeping some finetuning signal)
    reward_weighted = reward_bool * (0.9 + 0.1 * jnp.clip(1.-.5*dist_for_reward/env_params.line_done_cutoff, min=0., max=1.))
    return reward_weighted - penalty

def _update_env_state_from_action(rng_key: Key, env_state: EnvState, action: Action, env_params: EnvParams) -> EnvState:
    """
    NOTE: action expected in [-1, 1]^3; first 2 movement, third will be compared to 0 to determine if actually drawing
    Environment update could, in theory, be non-deterministic so introduce support here
    """
    old_position = env_state.position
    latest_stroke, new_position = _compute_stroke_and_pos_from_action(env_state, old_position, action, env_params)
    new_target_stroke_statuses, _ = _compute_new_stroke_statuses(latest_stroke, env_state, env_params)
    new_drawn_strokes = env_state.drawn_strokes.at[env_state.trial_step].set(latest_stroke)
    env_state = env_state.replace(drawn_strokes=new_drawn_strokes)
    env_state = env_state.replace(position=new_position)
    env_state = env_state.replace(target_strokes_status=new_target_stroke_statuses)
    env_state = env_state.replace(trial_step=env_state.trial_step + 1)
    return env_state


def _draw_rule_parameters(rule_init_key: Key, env_params: EnvParams) -> Tuple[Array, Array, Array, Array]:
    '''
    WARNING: this will be traced and made static, so cannot change ruleset at runtime !
    Fits envisioned use cases, but might come back into play later 
    '''

    def draw_rule_params_any(key):
        mode_key, ref_pos_key, ref_angle_key, decr_key = jax.random.split(key, 4)
        sort_mode = jax.random.randint(mode_key, (), minval=0, maxval=5)
        decreasing = jax.random.bernoulli(decr_key)
        ref_pos = jax.random.uniform(ref_pos_key, (2,), minval=0, maxval=1)
        ref_angle = jax.random.uniform(ref_angle_key, (), minval=0, maxval=2*jnp.pi)
        return sort_mode, decreasing, ref_pos, ref_angle

    def draw_rule_params_along_cardinal_directions(key):
        # Choose one of the 4 directions, no "decreasing" for now
        mode_key, ref_pos_key, ref_angle_key, decr_key = jax.random.split(key, 4)
        sort_mode = 0
        decreasing = 0
        ref_pos = jnp.zeros(2)
        ref_angle = jax.random.randint(ref_angle_key, (), minval=0, maxval=4) * jnp.pi / 2.
        return sort_mode, decreasing, ref_pos, ref_angle

    def draw_rule_params_parametric_directions(key):
        # Choose one of the 4 directions, no "decreasing" for now
        mode_key, ref_pos_key, ref_angle_key, decr_key = jax.random.split(key, 4)
        sort_mode = 0
        decreasing = 0
        ref_pos = jnp.zeros(2)
        ref_angle = jax.random.uniform(ref_angle_key, (), minval=0, maxval=2*jnp.pi)
        return sort_mode, decreasing, ref_pos, ref_angle

    def draw_rule_params_parametric_directions_with_decreasing(key):
        # Choose one of the 4 directions, no "decreasing" for now
        mode_key, ref_pos_key, ref_angle_key, decr_key = jax.random.split(key, 4)
        sort_mode = 0
        decreasing = jax.random.randint(ref_angle_key, (), minval=0, maxval=2)
        # decreasing = 1 #jax.random.randint(ref_angle_key, (), minval=0, maxval=2)
        ref_pos = jnp.zeros(2)
        ref_angle = jax.random.uniform(ref_angle_key, (), minval=0, maxval=2*jnp.pi)
        return sort_mode, decreasing, ref_pos, ref_angle

    func_map = {
        "any": draw_rule_params_any,
        "along_cardinal_directions": draw_rule_params_along_cardinal_directions,
        "along_parametric_directions" : draw_rule_params_parametric_directions,
        "along_parametric_directions_with_decreasing" : draw_rule_params_parametric_directions_with_decreasing,
    }

    return func_map[env_params.ruleset](rule_init_key)

def _initialize_env_state(env_init_key: Key, env_params: EnvParams) -> EnvState:
    """
    Draw starts/ends far from edges to avoid issues with too short lines (note that start/end are not ordered in any meaningful way)
    """
    n_strokes = env_params.num_target_strokes
    T = env_params.max_num_strokes
    stroke_min_length = env_params.stroke_min_length
    stroke_max_length = env_params.stroke_max_length
    stroke_thickness = env_params.thickness
    lengths_key, line_starts_key, start_pos_key, rule_init_key = jax.random.split(env_init_key, 4)
    target_starts = jax.random.uniform(line_starts_key, shape=(n_strokes, 2), minval=stroke_min_length, maxval=1.-stroke_min_length, dtype=jnp.float32) 
    uncorrected_moves = jax.random.uniform(lengths_key, shape=(n_strokes, 2), minval=-stroke_max_length, maxval=stroke_max_length, dtype=jnp.float32)
    target_ends = jnp.clip(target_starts + uncorrected_moves, 2*stroke_thickness, 1.-2*stroke_thickness)
    target_strokes = _canonicalize_strokes(jnp.concat([target_starts, target_ends], axis=-1)) # (S,2) (S,2) -> (S,4)
    drawn_strokes = -10*jnp.ones((T, 4), dtype=jnp.float32)
    target_strokes_status: TargetStrokesStatus = jnp.zeros(n_strokes, dtype=jnp.bool)
    start_pos = jax.random.uniform(start_pos_key, shape=(2,), minval=2*stroke_thickness, maxval=1.-2*stroke_thickness, dtype=jnp.float32)
    trial_step: TrialStep = jnp.array(0)

    sort_mode, decreasing, ref_point, ref_angle = _draw_rule_parameters(rule_init_key, env_params)

    # vmapped_reorderer = jax.vmap(_reorder_strokes, in_axes=(0,)*5)
    # reordered_target_strokes = vmapped_reorderer(target_strokes, sort_mode, ref_angle, ref_point, decreasing)
    reordered_target_strokes = _reorder_strokes(target_strokes, sort_mode, ref_angle, ref_point, decreasing)
    state = EnvState(target_strokes=reordered_target_strokes, drawn_strokes=drawn_strokes, position=start_pos, target_strokes_status=target_strokes_status, trial_step=trial_step,
                     sort_mode=sort_mode, decreasing=decreasing, ref_point=ref_point, ref_angle=ref_angle)
    return state

def batched_generate_obs(env_state: EnvStateBatch, env_params: EnvParams) -> CanvasBatch:
    """Observation generation should always be deterministic !"""
    def _unwrapped_logic(s):
        return _generate_observation(EnvState(**s.__dict__), env_params)
    return jax.vmap(_unwrapped_logic, in_axes=(0,))(env_state)

def offline_regenerate_observations_history(env_state: EnvStateHistory, env_params: EnvParams) -> CanvasHistory:
    """
    Observation generation should always be deterministic !
    NOTE: here we use a vmap also for time since env has already been replayed so no sequential dependency
    """
    def _unwrapped_logic(s):
        return batched_generate_obs(EnvStateBatch(**s.__dict__), env_params)
    return jax.vmap(_unwrapped_logic, in_axes=0)(env_state)

def _batched_initialize_env_states(rng_keys: KeyBatch, env_params: EnvParams) -> EnvStateBatch:
    def _unwrapped_logic(key, env_params):
        return _initialize_env_state(key, env_params)
    f = jax.vmap(_unwrapped_logic, in_axes=(0, None))
    tmp = f(rng_keys, env_params)
    return EnvStateBatch(**tmp.__dict__)

def _batched_update_states(rng_keys: KeyBatch, env_states: EnvStateBatch, actions: ActionBatch, env_params: EnvParams) -> EnvStateBatch:
    def _unwrapped_logic(key, state, action, env_params):
        state = _update_env_state_from_action(key, EnvState(**state.__dict__), action, env_params)
        return EnvStateBatch(**state.__dict__)
    state_batch = jax.vmap(_unwrapped_logic, in_axes=(0, 0, 0, None))(rng_keys, env_states, actions, env_params)
    return state_batch

def _batched_compute_reward_from_action(actions: ActionBatch, env_states: EnvStateBatch, env_params: EnvParams) -> RewardBatch:
    def _unwrapped_logic(action, state, env_params):
        reward = _compute_reward_from_action(cast(Action, action), EnvState(**state.__dict__), env_params)
        return cast(RewardBatch, reward)
    reward_batch = jax.vmap(_unwrapped_logic, in_axes=(0, 0, None))(actions, env_states, env_params)
    return reward_batch

def _wrap_policy_for_batch(policy: Policy) -> BatchedPolicy:
    def _unwrapped_logic(rng_key, policy_state, env_state, observation, env_params):
        return policy(rng_key, policy_state, EnvState(**env_state.__dict__), observation, env_params)
    f = jax.vmap(_unwrapped_logic, in_axes=(0, 0, 0, 0, None))
    def casted_f(rng_key: KeyBatch, policy_state: PolicyStateBatch, env_state: EnvStateBatch, observation: CanvasBatch, env_params: EnvParams):
        state, action = f(rng_key, policy_state, EnvState(**env_state.__dict__), observation, env_params)
        return cast(PolicyStateBatch, state), cast(ActionBatch, action)
    return casted_f

def _wrap_pol_init_for_batch(init_fn: PolicyStateInitializer) -> PolicyStateBatchInitializer:
    def _unwrapped_logic(rng_key):
        return init_fn(rng_key)
    f = jax.vmap(_unwrapped_logic, in_axes=(0,))
    def casted_f(rng_keys: KeyBatch) -> PolicyStateBatch:
        return cast(PolicyStateBatch, f(rng_keys))
    return casted_f

def off_policy_online_rollout(
                        # Traced
                        env_rng_key: Key, pol_rng_key: Key,   
                        # Can be traced, can be static
                        agent_policy: Policy, agent_state_init_fn: PolicyStateInitializer, 
                        teacher_policy: Policy, teacher_state_init_fn: PolicyStateInitializer, 
                        # Static
                        batch_size: int, env_params: EnvParams, visual=True
                        ) -> FullRollout:
    """
    Both agent and teacher policies are executed, actions and states tracked.
    Only teacher actually changes environment. If no need for teacher, use online_policy_rollout
    (an alias that uses same policy for both but returns same objects for simplicity)

    Initialization / rollout keys are shared between policies for compatibility with online_rollout
    They are separate from env to be able to explore agent stochasticity on a given env realization
    """
    T = env_params.max_num_strokes
    B = batch_size

    batched_agent_state_init = _wrap_pol_init_for_batch(agent_state_init_fn)
    batched_agent_policy = _wrap_policy_for_batch(agent_policy)

    batched_teacher_state_init = _wrap_pol_init_for_batch(teacher_state_init_fn)
    batched_teacher_policy = _wrap_policy_for_batch(teacher_policy)

    init_env_key, rollout_env_key = jax.random.split(env_rng_key, 2)
    initial_env_states = _batched_initialize_env_states(jax.random.split(init_env_key, B), env_params)

    init_pol_key, rollout_pol_key = jax.random.split(pol_rng_key, 2)
    init_pol_keys = jax.random.split(init_pol_key, B)
  
    initial_agent_states = batched_agent_state_init(init_pol_keys)
    initial_teacher_states = batched_teacher_state_init(init_pol_keys)

    initial_carry = BatchedStepCarry(agent_state=initial_agent_states, env_state=initial_env_states, teacher_state=initial_teacher_states)

    all_steps_pol_keys = jax.random.split(rollout_pol_key, T*B).reshape(T, B)
    all_steps_env_keys = jax.random.split(rollout_env_key, T*B).reshape(T, B)
    all_steps_keys: Key[Array, "T B 2"] = jnp.stack([all_steps_env_keys, all_steps_pol_keys], axis=-1)

    def grouped_batch_step(carry: BatchedStepCarry, step_keys: Key[Array, "B 2"]):
        env_step_keys, pol_step_keys = jnp.unstack(step_keys, axis=-1)
        old_env_states = carry.env_state
        old_agent_states = carry.agent_state
        old_teacher_states = carry.teacher_state
        if visual:
            old_obs: Float[Array, "B 3 H W"] = batched_generate_obs(old_env_states, env_params)
        else:
            # Not the most elegant, but for now it's good enough
            old_obs = jnp.zeros((B, 3, env_params.size, env_params.size))

        new_agent_states, agent_actions = batched_agent_policy(pol_step_keys, old_agent_states, old_env_states, old_obs, env_params)
        new_teacher_states, teacher_actions = batched_teacher_policy(pol_step_keys, old_teacher_states, old_env_states, old_obs, env_params)
        
        agent_rewards = _batched_compute_reward_from_action(agent_actions, old_env_states, env_params)
        teacher_rewards = _batched_compute_reward_from_action(teacher_actions, old_env_states, env_params)

        # Only update using the teacher action
        new_env_states = _batched_update_states(env_step_keys, old_env_states, teacher_actions, env_params)

        # Output uses pre-update states and obs to align correctly with actions / rewards
        output = RolloutStepOutput(
            env_state=old_env_states, 
            obs=old_obs, 
            agent_state=old_agent_states, 
            teacher_state=old_teacher_states, 
            agent_action=agent_actions, 
            teacher_action=teacher_actions, 
            agent_reward=agent_rewards, 
            teacher_reward=teacher_rewards, 
            )
        
        # Carry on the other hand needs the updated states !
        new_carry = carry.replace(agent_state=new_agent_states, teacher_state=new_teacher_states, env_state=new_env_states)
        
        return new_carry, output

    _, output_history = jax.lax.scan(grouped_batch_step, initial_carry, all_steps_keys)
    return cast(FullRollout, output_history)


def on_policy_online_rollout(
                        # Traced
                        env_rng_key: Key, pol_rng_key: Key,   
                        # Can be traced, can be static
                        policy: Policy, state_init_fn: PolicyStateInitializer, 
                        # Static
                        batch_size: int, env_params: EnvParams, visual=True
                        ) -> FullRollout:
    '''
    No teacher policy implemented as wrapper to teacher policy to avoid duplicated code
    At compilation time, this will get optimized so no double copies / computations should happen (see tests)
    Randomness for env not shared with agent to explore agent stochasticity on a given env realization
    '''
    return off_policy_online_rollout(env_rng_key, pol_rng_key, policy, state_init_fn, policy, state_init_fn, batch_size, env_params, visual=visual)

def offline_replay_actions(env_rng_key: Key[Array, ""], actions: ActionHistory, env_params: EnvParams) -> Tuple[EnvStateHistory, RewardHistory]:
    T = env_params.max_num_strokes
    B = actions.shape[1]

    init_env_key, rollout_env_key = jax.random.split(env_rng_key, 2)
    initial_env_states = _batched_initialize_env_states(jax.random.split(init_env_key, B), env_params)
    all_steps_env_keys = jax.random.split(rollout_env_key, T*B).reshape(T, B)
    step_inputs = {"keys": all_steps_env_keys, "actions": actions}

    def step(env_state: EnvStateBatch, step_input: dict[str, Array]):
        step_keys = step_input["keys"]
        step_actions = step_input["actions"]
        new_env_state = _batched_update_states(step_keys, env_state, step_actions, env_params)
        rewards = _batched_compute_reward_from_action(step_actions, env_state, env_params)
        step_outputs = {"env_state": env_state, "reward": rewards}
        return new_env_state, step_outputs

    _, output_history = jax.lax.scan(step, initial_env_states, step_inputs)
    env_state_history = cast(EnvStateHistory, output_history["env_state"])
    reward_history = cast(RewardHistory, output_history["reward"])
    return env_state_history, reward_history