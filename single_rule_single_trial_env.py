"""
Jax-based env: drawing done in jax via signed distance field, carry state based on small vectors only (not canvas to reduce memory bandwidth)
Compile the whole episode (for a single event, no batching) using scan, and only then vmap that
"""
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from functools import partial
from typing import Tuple, Optional

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CanvasParams:
    num_target_strokes: int = field(metadata=dict(static=True), default=4)
    max_num_strokes: int = field(metadata=dict(static=True), default=20)
    size: int = field(metadata=dict(static=True), default=128)
    stroke_min_length : float = field(metadata=dict(static=True), default=0.2)
    stroke_max_length : float = field(metadata=dict(static=True), default=0.4)
    # This ensures some gradient to help pinpoint center, but also visible
    thickness: float = field(metadata=dict(static=True), default=0.005)
    softness: float = field(metadata=dict(static=True), default=0.01)
    canvas_dtype: jnp.dtype = field(metadata=dict(static=True), default=jnp.float32)

@jax.tree_util.register_dataclass
@dataclass
class EnvState:
    ref_starts: jnp.array
    ref_ends: jnp.array
    drawn_starts: jnp.array
    drawn_ends: jnp.array
    position: jnp.array
    trial_step: int
    terminated: bool = False

@jax.tree_util.register_dataclass
@dataclass
class PolicyState:
    internal_state: Optional[jnp.array] = None

@jax.tree_util.register_dataclass
@dataclass
class StepCarry:
    policy_state: PolicyState
    env_state: EnvState

@jax.tree_util.register_dataclass
@dataclass
class Action:
    movement: jnp.array
    pressure: jnp.array

@jax.tree_util.register_dataclass
@dataclass
class StepOutput:
    env_state: EnvState
    action: Action
    obs: jnp.array

from typing import Protocol

class Policy(Protocol):
    def __call__(self, policy_state: PolicyState, env_state: EnvState, policy_step_rng_key: jax.random.key, canvas_params: CanvasParams) -> Tuple[PolicyState,Action]:
        ...

def dist_to_segments(points: jnp.array, starts: jnp.array, ends: jnp.array):
    # points expected to be canvas_params.coord_grid[:, :, None, :], size (H, W, 2)
    # starts and ends expected (n_strokes, 2)
    # All computations done in float32 since no need for precision
    line_vecs = ends-starts # (n_strokes, 2)
    line_square_norms = jnp.sum(line_vecs**2, axis=-1) + 1e-6 # (n_strokes,)
    vec_to_start = points[:,:,None,:] - starts[None, None, :, :] # (H, W, n_strokes, 2); points need expand across n_strokes, starts across frame axes

    # Start by finding the projection on the line, then the segment through clipping
    proj_signed_length = jnp.sum(vec_to_start * line_vecs[None, None, :, :], axis=-1) / line_square_norms[None, None, :] # (H, W, n_strokes,)

    # proj_signed_length (H, W, n_strokes,) needs expanding on the x/y dim at the end only, line_vecs across HW
    projection_on_segment = starts[None, None, :, :] + jnp.clip(proj_signed_length[:, :, :, None], 0, 1) * line_vecs[None, None, :, :] # (H, W, n_strokes, 2)

    dist_to_segments = jnp.linalg.norm(points[:, :, None, :] - projection_on_segment, axis=-1) # (H, W, n_strokes)
    return dist_to_segments


def draw_strokes_single_canvas(starts: jnp.array, ends: jnp.array, canvas_params: CanvasParams):
    """
    Can be used for 1 or many strokes at once, but expects an array of starts / ends (n_strokes, 2)
    Returns a full canvas at once to avoid any copies/bottlenecks
    Will need to integrate it into full env 
    """
    points = jnp.indices((canvas_params.size, canvas_params.size), dtype=jnp.float32).transpose(1, 2, 0) / canvas_params.size
    
    dists = dist_to_segments(points, starts, ends) # H, W, n_strokes

    # Now, intensity based on min distance to any stroke
    min_dist = jnp.min(dists, axis=-1) 
    # canvas = jnp.clip((canvas_params.thickness + canvas_params.softness - min_dist) / canvas_params.softness, 0.0, 1.0)
    # canvas = jnp.clip(min_dist, 0., 1.) 
    # canvas = 1.0 - jnp.clip((min_dist - canvas_params.thickness) / canvas_params.softness, 0.0, 1.0)
    canvas = jnp.clip(1. - jnp.clip((min_dist - canvas_params.thickness) / (canvas_params.softness+1e-6), 0.0, .99), 0.01, .99)
    return canvas

def draw_point_single_canvas(point: jnp.array, canvas_params: CanvasParams):
    """
    Basically reuse function for strokes, but since it makes no sense to add dim of 1 to position do some wrapping instead 
    Reducing the segment to a point works thanks to regularizer
    """
    starts = point[None, :] 
    ends = point[None, :]
    return draw_strokes_single_canvas(starts, ends, canvas_params)
    
def generate_observation(env_state: EnvState, canvas_params: CanvasParams):
    """
    Should always be called with full-length line arrays to compile only once
    For "not-yet-taken" steps, simply put start=end=-10 
    """
    target_canvas = draw_strokes_single_canvas(env_state.ref_starts, env_state.ref_ends, canvas_params)

    draw_canvas = draw_strokes_single_canvas(env_state.drawn_starts, env_state.drawn_ends, canvas_params)
    # draw_canvas = draw_strokes_single_canvas(env_state.ref_starts, env_state.ref_ends, canvas_params)

    pos_canvas = draw_point_single_canvas(env_state.position, canvas_params)
    return jnp.stack((pos_canvas, draw_canvas, target_canvas), axis=0)

def initialize_env_state(rng_key: jax.random.key, canvas_params: CanvasParams):
    """
    Need to draw the starts/ends cleanly; to ensure no issue with oob, draw start far from edges 
    Could rework it to have vmpa here too rather than stack, but keep it for now since logic could diverge between each once we add reward
    """
    n_strokes = canvas_params.num_target_strokes
    T = canvas_params.max_num_strokes
    stroke_min_length = canvas_params.stroke_min_length
    stroke_max_length = canvas_params.stroke_max_length
    stroke_thickness = canvas_params.thickness

    lengths_key, line_starts_key, start_pos_key = jax.random.split(rng_key, 3)
    ref_starts = jax.random.uniform(line_starts_key, shape=(n_strokes, 2), minval=stroke_min_length, maxval=1.-stroke_min_length, dtype=jnp.float32) 
    lines_vecs = jax.random.uniform(lengths_key, shape=(n_strokes, 2), minval=-stroke_max_length, maxval=stroke_max_length, dtype=jnp.float32)
    ref_ends = jnp.clip(ref_starts + lines_vecs, 2*stroke_thickness, 1.-2*stroke_thickness)
    start_pos = jax.random.uniform(start_pos_key, shape=(2,), minval=2*stroke_thickness, maxval=1.-2*stroke_thickness, dtype=jnp.float32)

    state = EnvState(ref_starts=ref_starts, 
                        ref_ends=ref_ends, 
                        drawn_starts=-10*jnp.ones((T, 2), dtype=jnp.float32), 
                        drawn_ends=-10*jnp.ones((T, 2), dtype=jnp.float32), 
                        position=start_pos, 
                        trial_step=0)

    return state

def random_agent_policy(policy_state: PolicyState, env_state: EnvState, policy_step_rng_key: jax.random.key, canvas_params: CanvasParams) -> Tuple[PolicyState, Action]:
    """
    Here for testing / interface clarification only.
    TODO: implement oracle policy (hard)
    """
    stroke_max_length = canvas_params.stroke_max_length
    movement_key, pressure_key = jax.random.split(policy_step_rng_key)
    movement = jax.random.uniform(movement_key, shape=(2,), minval=-stroke_max_length, maxval=stroke_max_length, dtype=jnp.float32)
    pressure = jax.random.bernoulli(pressure_key, p=0.5, shape=(1,))
    new_policy_state = policy_state
    return new_policy_state, Action(movement=movement, pressure=pressure)

def update_env_state_from_action(env_state: EnvState, action: Action, canvas_params: CanvasParams) -> EnvState:
    """
    This updates the state dataclass in-place, should be very fast so no worries; heavy lifting is in drawing, this comes later in "draw_observations"
    """
    old_position = env_state.position
    new_position = jnp.clip(env_state.position + action.movement, canvas_params.thickness, 1.-canvas_params.thickness) 

    drawn_starts_step = env_state.drawn_starts[env_state.trial_step]
    drawn_ends_step = env_state.drawn_ends[env_state.trial_step]
    # This will trigger a copy, should be ok since very small and optimized away in jit (hopefully)
    # Use standard boolean switch without branching
    new_starts = env_state.drawn_starts.at[env_state.trial_step].set((1-action.pressure) * drawn_starts_step + action.pressure * old_position)
    new_ends = env_state.drawn_ends.at[env_state.trial_step].set((1-action.pressure) * drawn_ends_step + action.pressure * new_position)

    new_trial_step = env_state.trial_step + 1
    env_state.drawn_ends = new_ends
    env_state.drawn_starts = new_starts
    env_state.position = new_position
    env_state.trial_step = new_trial_step


    return env_state

@partial(jax.jit, static_argnums=(1,2,))
def rollout_one_trial(trial_rng_key: jax.random.key, policy_fn: Policy, canvas_params: CanvasParams):
    """
    Does an entire trial rollout, from initialization to closure.
    """
    initial_key, rollout_key = jax.random.split(trial_rng_key)
    initial_state = initialize_env_state(initial_key, canvas_params)
    initial_carry = StepCarry(env_state=initial_state, policy_state=PolicyState(internal_state=None))

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
        batch_size = 64
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
            batch_observations = (256* np.asarray(batch_history.obs, dtype=np.float64)).astype(np.uint8).transpose(0,1,3,4,2)
            print(batch_observations.shape)
            logging.critical(f"Batch {batch} done in {times[-1]:.5f}s")
            for b_idx in range(5):
                images = [Image.fromarray(batch_observations[b_idx,t]) for t in range(batch_observations.shape[1])]
                images[0].save(save_path / f'sanity_check_batch_{batch}_trajectory_{b_idx}.gif', save_all=True,
                append_images=images[1:], # append rest of the images
                duration=1000, # in milliseconds
                loop=0)

    test_rollout()
