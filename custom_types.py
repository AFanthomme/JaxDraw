import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass, field
from typing import Tuple, Protocol, Optional
from jaxtyping import Float, Array, Key

# These will be used by pylance and should help make things cleaner

# Heavy, not to be carried, image types
type FullCanvas = Float[Array, "H W 3"]
"""Float[Array, "H W 3"], 3 is for RGB, aka pos, drawn, target; Value range [0, 1[ for correct image conversion"""

type SingleFrame = Float[Array, "H W"]
"""Float[Array, "H W"], Value range [0, 1[ for correct image conversion"""

type FrameCoordinates = Float[Array, "H W 2"]
"""Float[Array, "H W 2"], Value range [0, 1], the xy coord vector for each of the pixels in the frame"""

type TargetStrokesDistanceFields = Float[Array, "H W S"]
"""Float[Array, "H W S"], Value range [0, 1], the distance to each target stroke for each of the pixels in the frame"""

# Meaningful array shapes / value ranges
type CoordinateVector = Float[Array, "2"]
"""Float[Array, "2"]: Value range [0, 1] to stay within frame"""

type MovementVector = Float[Array, "2"]
"""Float[Array, "2"], Value range [-1, 1]; can lead to leaving frame, env is responsible for clipping"""

type PressureValue = Float[Array, "1"]
"""Float[Array, "1"], Value range [-1, 1]; negative means not drawing"""

type TargetStrokesCollection = Float[Array, "S 4"]
"""Float[Array, "S 4"], Each of the S strokes is x_1, y_1, x_2, y_2; values in [0, 1]"""

type DrawnStrokesCollection = Float[Array, "T 4"]
"""Float[Array, "T 4"], T is max trial length, ie max strokes possible\n each of the T strokes is x_1, y_1, x_2, y_2; values in [0, 1]"""

type StrokesCollection = DrawnStrokesCollection | TargetStrokesCollection
"""Float[Array, "S|T 4"], x_1, y_1, x_2, y_2, first dim shape depends whether those are targets or drawn"""

type TargetCoverageQuality = Float[Array, "S"]
"""Float[Array, "S"], Gives a score (in [0,1]) to how much a given TargetStroke corresponds to at least one TargetStroke"""


type DrawnCoverageQuality = Float[Array, "T"]
"""Float[Array, "T"], Gives a score (in [0,1]) to how much each DrawnStroke corresponds to at least one DrawnStroke"""

type TrialStep = int
"""int, Number of elapsed timesteps within one trial (not to be confused with BlockStep or TrainStep); values in {0..T-1}"""

type PolicyState = Optional[Float[Array, "H"]]
"""Float[Array, "H"], Vector hidden state for the policy; unused for now but will come with recurrent policies so type it in"""



# RNG keys, not sure very useful but might as well keep track of them too
type EnvInitRngKey = Key
"""RNG key for the initialization of the environment state"""
type PolicyStepRngKey = Key
"""RNG key for the current step of the policy"""
type EnvStepRngKey = Key
"""RNG key for the current step of the environment; useful if need reset (then used as init key) or to add agent-independent noise"""
type TrialRngKey = Key
"""RNG key for a full trial, ie T steps of env/policy; to be split into 2T keys (env, policy for each step)"""
type BatchRngKey = Key
"""RNG key for a batch of B trials, to be split into B keys (one for each trial in the batch)"""

# Future: 
# type BlockRngKey = Annotated[Key, "RNG key for a full block, ie G trials of length T (to be split into G keys, one for each trial)"] 


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CanvasParams:
    num_target_strokes: int = field(metadata=dict(static=True), default=4)
    max_num_strokes: int = field(metadata=dict(static=True), default=20)
    size: int = field(metadata=dict(static=True), default=128)
    stroke_min_length: float = field(metadata=dict(static=True), default=0.2)
    stroke_max_length: float = field(metadata=dict(static=True), default=0.4)
    # This ensures some gradient to help pinpoint center, but also visible
    thickness: float = field(metadata=dict(static=True), default=0.005)
    softness: float = field(metadata=dict(static=True), default=0.01)
    canvas_dtype: jnp.dtype = field(metadata=dict(static=True), default=jnp.float32)

@jax.tree_util.register_dataclass
@dataclass
class EnvState:
    '''
    Everything done to avoid carrying visual state between timesteps for reduced memory bandwidth
    NOTE: For now, keep rng state out of EnvState for more flexibility, might change later
    '''
    target_strokes: TargetStrokesCollection
    drawn_strokes: DrawnStrokesCollection
    position: CoordinateVector
    target_qualities: TargetCoverageQuality
    drawn_qualities: DrawnCoverageQuality
    trial_step: TrialStep

@jax.tree_util.register_dataclass
@dataclass
class StepCarry:
    policy_state: PolicyState
    env_state: EnvState

@jax.tree_util.register_dataclass
@dataclass
class Action:
    movement: MovementVector
    pressure: PressureValue

@jax.tree_util.register_dataclass
@dataclass
class StepOutput:
    env_state: EnvState
    action: Action
    obs: FullCanvas

class Policy(Protocol):
    def __call__(self, policy_state: PolicyState, env_state: EnvState, policy_step_rng_key: PolicyStepRngKey, canvas_params: CanvasParams) -> Tuple[PolicyState,Action]:
        ...
