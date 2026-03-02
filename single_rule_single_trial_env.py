"""
Jax-based env: drawing done in jax via signed distance field, carry state based on small vectors only (not canvas to reduce memory bandwidth)
Compile the whole episode (for a single event, no batching) using scan, and only then vmap that
"""
import jax
import numpy as np
import jax.numpy as jnp
from custom_types import *

def initialize_env_state(env_init_key: EnvInitRngKey, canvas_params: CanvasParams) -> EnvState:
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

    # TODO: modify TargetStrokesCollection so that start/end are always ordered in the same way? Would make comparisons later easier,
    # but won't be able to do it for DrawnStrokes so might be useless anyway
    target_strokes: TargetStrokesCollection = jnp.concat([target_starts, target_ends], axis=-1) # (S,2) (S,2) -> (S,4)
    drawn_strokes: DrawnStrokesCollection = -10*jnp.ones((T, 4), dtype=jnp.float32)
    target_qualities: TargetCoverageQuality = jnp.zeros(n_strokes)
    drawn_qualities: DrawnCoverageQuality = jnp.zeros(T)

    start_pos: CoordinateVector = jax.random.uniform(start_pos_key, shape=(2,), minval=2*stroke_thickness, maxval=1.-2*stroke_thickness, dtype=jnp.float32)
    trial_step: TrialStep = 0

    state = EnvState(target_strokes=target_strokes, drawn_strokes=drawn_strokes, position=start_pos, target_qualities=target_qualities, drawn_qualities=drawn_qualities, trial_step=trial_step)

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


# def update_line_qualities(env_state: EnvState):
#     """
#     For now, compute this based solely on start-end; could probably compute the dice implicitly 
#     (would be better once we use more complex shapes?) but would require individual frames in obs
#     making it significantly more computationally expensive
#     """

def update_env_state_from_action(env_state: EnvState, action: Action, canvas_params: CanvasParams) -> EnvState:
    """
    Basic flow control without branching to decide whether we update the DrawnStrokes or not
    """
    old_position = env_state.position
    new_position = jnp.clip(env_state.position + action.movement, canvas_params.thickness, 1.-canvas_params.thickness) 
    proposed_line = jnp.concat((old_position, new_position), axis=-1)
    default_drawn_line_state: Float[Array, "4"] = -10*jnp.ones(4, dtype=jnp.float32)
    new_line_state = (1-action.pressure) * default_drawn_line_state + action.pressure * proposed_line

    # This might trigger a copy, or get XLA-optimized away; in any case, those are all small so should not be disastrous
    env_state.drawn_strokes = env_state.drawn_strokes.at[env_state.trial_step].set(new_line_state)
    env_state.position = new_position
    env_state.trial_step = env_state.trial_step + 1

    return env_state


def random_agent_policy(policy_state: PolicyState, env_state: EnvState, policy_step_rng_key: PolicyStepRngKey, canvas_params: CanvasParams) -> Tuple[PolicyState, Action]:
    """
    Here for testing / interface clarification only.
    TODO: implement oracle policy
    """
    stroke_max_length = canvas_params.stroke_max_length
    movement_key, pressure_key = jax.random.split(policy_step_rng_key)
    movement = jax.random.uniform(movement_key, shape=(2,), minval=-stroke_max_length, maxval=stroke_max_length, dtype=jnp.float32)
    pressure = jax.random.bernoulli(pressure_key, p=0.5, shape=(1,))
    new_policy_state = policy_state
    return new_policy_state, Action(movement=movement, pressure=pressure)




@partial(jax.jit, static_argnums=(1,2,))
def rollout_one_trial(trial_rng_key: TrialRngKey, policy_fn: Policy, canvas_params: CanvasParams):
    """
    Does an entire trial rollout, from initialization to closure.
    """
    initial_key, rollout_key = jax.random.split(trial_rng_key)
    initial_state = initialize_env_state(initial_key, canvas_params)
    initial_carry = StepCarry(env_state=initial_state, policy_state=None)

    steps_keys = jax.random.split(rollout_key, canvas_params.max_num_strokes)

    # NOTE: this is the only valid signature for jax.scan body_fn: carry, step_input -> new_carry, output
    # TODO: handle terminated episodes? Since will compute anyway, not sure it makes sense
    def step_fn(carry: StepCarry, steps_keys):
        env_state = carry.env_state
        policy_state = carry.policy_state
        obs = generate_observation(env_state, canvas_params)
        new_policy_state, action = policy_fn(policy_state, env_state, steps_keys, canvas_params=canvas_params)
        new_env_state = update_env_state_from_action(env_state, action, canvas_params)
        new_carry = StepCarry(policy_state=new_policy_state, env_state=env_state)
        return new_carry, StepOutput(env_state=new_env_state, action=action, obs=obs)

    final_state, output_history = jax.lax.scan(step_fn, initial_carry, steps_keys)
    return output_history

if __name__ == '__main__':
    import time
    from PIL import Image
    import os
    import logging
    from pathlib import Path
    def test_rollout():
        logging.critical("Starting rollout tests")
        batch_size = 128
        default_canvas_params = CanvasParams()
        save_path = Path('tests/single_rule_single_trial/observation_gifs')
        save_path.mkdir(exist_ok=True, parents=True)

        batch_loop_fn = jax.vmap(rollout_one_trial, in_axes=(0, None, None))
        times = []
        for batch in range(10):
            batch_rng_key = jax.random.key(777+batch)
            subkeys = jax.random.split(batch_rng_key, batch_size)
            tic = time.time()
            batch_history = batch_loop_fn(subkeys, random_agent_policy, default_canvas_params)
            times.append(time.time()-tic)
            batch_observations = (256* np.asarray(batch_history.obs, dtype=np.float64)).astype(np.uint8)
            batch_observations = batch_observations.transpose(0,1,3,4,2)
            logging.critical(f"Batch {batch} done in {times[-1]:.5f}s")
            for b_idx in range(5):
                images = [Image.fromarray(batch_observations[b_idx,t]) for t in range(batch_observations.shape[1])]
                images[0].save(save_path / f'sanity_check_batch_{batch}_trajectory_{b_idx}.gif', save_all=True,
                append_images=images[1:], # append rest of the images
                duration=1000, # in milliseconds
                loop=0)

    test_rollout()
