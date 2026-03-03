"""
Jax-based env: drawing done in jax via signed distance field, carry state based on small vectors only (not canvas to reduce memory bandwidth)
Compile the whole episode (for a single event, no batching) using scan, and only then vmap that
"""
import jax
import numpy as np
import jax.numpy as jnp
from custom_types import *
import time
from PIL import Image
import logging
from pathlib import Path

def canonicalize_strokes(strokes: StrokesCollection|Stroke) -> StrokesCollection|Stroke:
    """
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

def initialize_env_state(env_init_key: Key, canvas_params: CanvasParams) -> EnvState:
    """
    Draw starts/ends far from edges to avoid issues with too short lines (note that start/end are not ordered in any meaningful way)
    """
    n_strokes = canvas_params.num_target_strokes
    T = canvas_params.max_num_strokes
    stroke_min_length = canvas_params.stroke_min_length
    stroke_max_length = canvas_params.stroke_max_length
    stroke_thickness = canvas_params.thickness

    lengths_key, line_starts_key, start_pos_key = jax.random.split(env_init_key, 3)

    target_starts: CoordinateVector = jax.random.uniform(line_starts_key, shape=(n_strokes, 2), minval=stroke_min_length, maxval=1.-stroke_min_length, dtype=jnp.float32) 
    uncorrected_moves: MovementVector = jax.random.uniform(lengths_key, shape=(n_strokes, 2), minval=-stroke_max_length, maxval=stroke_max_length, dtype=jnp.float32)
    target_ends: CoordinateVector = jnp.clip(target_starts + uncorrected_moves, 2*stroke_thickness, 1.-2*stroke_thickness)

    target_strokes: TargetStrokesCollection = canonicalize_strokes(jnp.concat([target_starts, target_ends], axis=-1)) # (S,2) (S,2) -> (S,4)
    drawn_strokes: DrawnStrokesCollection = -10*jnp.ones((T, 4), dtype=jnp.float32)
    target_strokes_status: TargetStrokesStatus = jnp.zeros(n_strokes, dtype=jnp.bool)

    start_pos: CoordinateVector = jax.random.uniform(start_pos_key, shape=(2,), minval=2*stroke_thickness, maxval=1.-2*stroke_thickness, dtype=jnp.float32)
    trial_step: TrialStep = 0
    step_reward: StepReward = 0.

    state = EnvState(target_strokes=target_strokes, drawn_strokes=drawn_strokes, position=start_pos, target_strokes_status=target_strokes_status, step_reward=step_reward, trial_step=trial_step)

    return state

def dist_to_segments(frame_coordinates: FrameCoordinates, strokes: StrokesCollection) -> TargetStrokesDistanceFields:
    """
    All computations done in float32 since no need for precision
    """
    ends, starts = jnp.split(strokes, 2, axis=-1) 
    line_vecs = ends-starts # (n_strokes, 2)
    line_square_norms = jnp.sum(line_vecs**2, axis=-1) + 1e-6 # (n_strokes,)
    vec_to_start = frame_coordinates[:,:,None,:] - starts[None, None, :, :] # (H, W, n_strokes, 2); points need expand across n_strokes, starts across frame axes

    # Start by finding the projection on the line, then the segment through clipping
    proj_signed_length = jnp.sum(vec_to_start * line_vecs[None, None, :, :], axis=-1) / line_square_norms[None, None, :] # (H, W, n_strokes,)

    # proj_signed_length (H, W, n_strokes,) needs expanding on the x/y dim at the end only, line_vecs across HW
    projection_on_segment = starts[None, None, :, :] + jnp.clip(proj_signed_length[:, :, :, None], 0, 1) * line_vecs[None, None, :, :] # (H, W, n_strokes, 2)

    dist_to_segments: TargetStrokesDistanceFields = jnp.linalg.norm(frame_coordinates[:, :, None, :] - projection_on_segment, axis=-1) 
    return dist_to_segments

def draw_strokes_single_canvas(strokes: StrokesCollection, canvas_params: CanvasParams) -> SingleFrame:
    frame_coordinates: FrameCoordinates = jnp.indices((canvas_params.size, canvas_params.size), dtype=jnp.float32).transpose(1, 2, 0) / canvas_params.size
    dists: TargetStrokesDistanceFields = dist_to_segments(frame_coordinates, strokes) 
    frame_pixels: SingleFrame = jnp.clip(1. - jnp.clip((jnp.min(dists, axis=-1) - canvas_params.thickness) / (canvas_params.softness+1e-6), 0.0, .99), 0.0, .99) 
    return frame_pixels

def draw_point_single_canvas(coordinate_vector: CoordinateVector, canvas_params: CanvasParams) -> SingleFrame:
    """
    Reducing the segment to a point allows to reuse strokes draw call, so do that
    """
    point_as_stroke_collection: StrokesCollection = jnp.tile(coordinate_vector, 2)[None, :]
    return draw_strokes_single_canvas(point_as_stroke_collection, canvas_params) 
    
def generate_observation(env_state: EnvState, canvas_params: CanvasParams) -> FullCanvas:
    """
    Generate the visual observation; internal state update kept separate since more lightweight
    so could be used in "dreaming" loop without having to generate observations at each step
    """
    target_canvas = draw_strokes_single_canvas(env_state.target_strokes, canvas_params)
    draw_canvas = draw_strokes_single_canvas(env_state.drawn_strokes, canvas_params)
    pos_canvas = draw_point_single_canvas(env_state.position, canvas_params)

    return jnp.stack((pos_canvas, draw_canvas, target_canvas), axis=0)

def update_stroke_status(env_state: EnvState, canvas_params: CanvasParams) -> EnvState:
    """
    For now, compute this based solely on start-end; could probably compute the dice implicitly 
    (would be better once we use more complex shapes or do more granular movement) but 
    would require individual frames in obs making it significantly more computationally expensive.

    NOTE: Strokes are expected to already be in lexicographic order, but not used to avoid brittle edges
    """
    latest_stroke = env_state.drawn_strokes[env_state.trial_step]

    # this does not work yet
    def is_valid_match(test_stroke: Stroke, target_stroke: Stroke):
        """
        Checks if test_stroke matches ref_stroke based on endpoint proximity
        Tests all pairings of endpoints in parallel, and ensures not both test endpoints are close to the same target
        """
        threshold = canvas_params.quality_max_pos_dif

        # Compute all 4 pairwise squared distances via broadcasting
        t_pts = test_stroke.reshape(2, 2)
        r_pts = target_stroke.reshape(2, 2)
        diffs = t_pts[:, None, :] - r_pts[None, :, :]
        # Use L_infty distance for easier noise manipulation
        dist = jnp.max(jnp.abs(diffs), axis=-1)
        
        # Check that each test endpoint is within threshold of SOME ref endpoint
        min_dists_sq = jnp.min(dist, axis=1)
        within_threshold = jnp.all(min_dists_sq < threshold)
        
        # Check that both test endpoints are not closest to the same ref endpoint
        l2_dist = jnp.sum((diffs**2), axis=-1)
        closest_ref_indices = jnp.argmin(l2_dist, axis=1)
        distinct_endpoints = closest_ref_indices[0] != closest_ref_indices[1]
        
        return within_threshold & distinct_endpoints
    
    current_matches = jax.vmap(is_valid_match, in_axes=(None, 0))(latest_stroke, env_state.target_strokes)
    old_status = env_state.target_strokes_status

    # Get a reward only if the match is really new
    step_reward = jnp.any(current_matches & (~old_status)).astype(jnp.float32)

    # Update the line status
    new_status = current_matches | old_status
    env_state = env_state.replace(target_strokes_status=new_status)
    env_state = env_state.replace(step_reward=step_reward)

    return env_state

def update_env_state_from_action(env_state: EnvState, action: Action, canvas_params: CanvasParams) -> EnvState:
    """
    Basic flow control without branching to decide whether we update the DrawnStrokes or not
    """
    old_position = env_state.position
    new_position = jnp.clip(env_state.position + action[:2], canvas_params.thickness, 1.-canvas_params.thickness) 
    proposed_line = jnp.concat((old_position, new_position), axis=-1)
    default_drawn_line_state: Float[Array, "4"] = -10*jnp.ones(4, dtype=jnp.float32)
    new_line_state = (1-action[2]) * default_drawn_line_state + action[2] * proposed_line
    new_drawn_strokes = env_state.drawn_strokes.at[env_state.trial_step].set(canonicalize_strokes(new_line_state))
    env_state = env_state.replace(drawn_strokes=new_drawn_strokes)
    env_state = env_state.replace(position=new_position)

    env_state = update_stroke_status(env_state, canvas_params)
    env_state = env_state.replace(trial_step=env_state.trial_step + 1)

    return env_state

def random_agent_policy(policy_step_rng_key: Key, canvas_params: CanvasParams, agent_state: Optional[AgentState], env_state: EnvState, observation: Optional[FullCanvas]) -> Tuple[AgentState, Action]:
    """
    Here for testing / interface clarification only.
    """
    stroke_max_length = canvas_params.stroke_max_length
    movement_key, pressure_key = jax.random.split(policy_step_rng_key)
    movement = jax.random.uniform(movement_key, shape=(2,), minval=-stroke_max_length, maxval=stroke_max_length, dtype=jnp.float32)
    pressure = jax.random.bernoulli(pressure_key, p=0.5, shape=(1,))

    # NOTE: This is where policy would get updated
    new_policy_state = agent_state
    return new_policy_state, jnp.concat((movement, pressure))

def oracle_policy(policy_step_rng_key: Key, canvas_params: CanvasParams, policy_state: Optional[AgentState], env_state: EnvState, observation: Optional[FullCanvas]) -> Tuple[AgentState, Action]:
    """
    Having access to env_state is cheating, but that's what Oracles are all about

    Can probably do cleaner/better
    """
    # Make a (2S, 2) list of strokes
    endpoints = jnp.concat(jnp.split(env_state.target_strokes, 2, -1), axis=0)
    status = jnp.tile(env_state.target_strokes_status, 2)
    line_ids = jnp.tile(jnp.arange(canvas_params.num_target_strokes), 2)
    position: CoordinateVector = env_state.position

    dists = jnp.sum((position[None, :] - endpoints)**2, -1)
    filtered_dists = jnp.where(status, 2**10, dists) # put done lines' endpoints at infinity
    min_dist = jnp.min(filtered_dists)
    # Will draw only if we're already at an endpoint and not all lines done; not need to check all lines done since min_dist would go to infty anyway
    # in that case, move to the far_point (which is a pain to determine without branching)
    pressure = (min_dist < canvas_params.quality_max_pos_dif**2)

    target_line_id = line_ids[jnp.argmin(filtered_dists)]
    point1, point2 = jnp.split(env_state.target_strokes[target_line_id], 2)
    dist1, dist2 = jnp.sum((position-point1)**2), jnp.sum((position-point2)**2)
    close_point = jnp.where(dist1<=dist2, point1, point2)
    far_point = jnp.where(dist1<=dist2, point2, point1)
    target_endpoint = jnp.where(pressure, far_point, close_point)

    # Don't move if all lines already done
    movement = jnp.where(jnp.all(status), jnp.zeros(2, dtype=jnp.float32), target_endpoint - position)

    return None, jnp.concat((movement, jnp.array([pressure])))

def noisy_oracle_policy(policy_step_rng_key: Key, canvas_params: CanvasParams, policy_state: Optional[AgentState], env_state: EnvState, observation: Optional[FullCanvas]) -> Tuple[AgentState, Action]:
    """
    Allows for slightly imperfect trajectories
    """
    _, oracle_actions = oracle_policy(policy_step_rng_key, canvas_params, policy_state, env_state, observation)
    action_noise = jnp.array([0.9, 0.9, 0.]) * canvas_params.quality_max_pos_dif * jax.random.uniform(policy_step_rng_key, shape=(3,))
    return None, oracle_actions + action_noise

def rollout_one_evaluation_trial(trial_rng_key: Key, agent_policy: Policy, agent_state_init_fn: AgentStateInitializer, oracle_policy: Policy, canvas_params: CanvasParams, oracle_forcing: Bool=False):
    """
    Does an entire trial rollout, from initialization to closure.
    """
    initial_key, rollout_key = jax.random.split(trial_rng_key)
    env_init_key, agent_init_key = jax.random.split(initial_key)
    initial_env_state = initialize_env_state(env_init_key, canvas_params)
    initial_agent_state = agent_state_init_fn(agent_init_key)
    initial_carry = StepCarry(env_state=initial_env_state, agent_state=initial_agent_state)

    steps_keys = jax.random.split(rollout_key, canvas_params.max_num_strokes)

    def step_fn(carry: StepCarry, steps_keys):
        obs = generate_observation(carry.env_state, canvas_params)
        # Not forcing env_state to None here because we can use oracle as agent too 
        new_agent_state, agent_action = agent_policy(steps_keys, canvas_params, carry.agent_state, carry.env_state, obs)
        _, oracle_action = oracle_policy(steps_keys, canvas_params, None, carry.env_state, obs)
        new_env_state = update_env_state_from_action(carry.env_state, (1.-oracle_forcing)*agent_action+ oracle_forcing*oracle_action, canvas_params)
        new_carry = carry.replace(agent_state=new_agent_state, env_state=new_env_state)
        output = StepOutput(env_state=new_env_state, agent_state=new_agent_state, agent_action=agent_action, oracle_action=oracle_action, obs=obs)
        return new_carry, output

    _, output_history = jax.lax.scan(step_fn, initial_carry, steps_keys)
    return output_history

def perform_tests(batch_size=32, n_reps=10, save_path = Path('tests/single_rule_single_trial/observation_gifs')):
    save_path.mkdir(exist_ok=True, parents=True)
    logging.critical("Starting test rollouts")
    default_canvas_params = CanvasParams()
    batch_loop_fn = jax.vmap(rollout_one_evaluation_trial, in_axes=(0, None, None, None, None, None))

    jitted_batch_fn = jax.jit(
        batch_loop_fn, 
        static_argnums=(1, 2, 3, 4, 5)
    )

    def agent_state_dummy_init(key: Key) -> AgentState:
        return None

    times = []
    for batch in range(n_reps):
        batch_rng_key = jax.random.key(777+batch)
        subkeys = jax.random.split(batch_rng_key, batch_size)
        tic = time.time()
        batch_history = jitted_batch_fn(subkeys, noisy_oracle_policy, agent_state_dummy_init, oracle_policy, default_canvas_params, False)
        times.append(time.time()-tic)
        batch_observations = (256* np.asarray(batch_history.obs, dtype=np.float64)).astype(np.uint8)
        batch_observations = batch_observations.transpose(0,1,3,4,2)
        logging.critical(f"Batch {batch} done in {times[-1]:.5f}s, total rewards: {np.asarray(batch_history.env_state.step_reward, dtype=np.float64).sum()}")
        for b_idx in range(5):
            # TODO: add visualization for the reward
            images = [Image.fromarray(batch_observations[b_idx,t]) for t in range(batch_observations.shape[1])]
            images[0].save(save_path / f'sanity_check_batch_{batch}_trajectory_{b_idx}.gif', save_all=True,
            append_images=images[1:], # append rest of the images
            duration=1000, # in milliseconds
            loop=0)

if __name__ == '__main__':
    perform_tests()
