import jax
import jax.numpy as jnp
import chex 
from typing import Tuple, Protocol, Optional, TypeVar, Type, dataclass_transform
from jaxtyping import Float, Array, Key, Bool

T = TypeVar("T")

@dataclass_transform()
class JaxDataclass:
    """Base class that fixes Pylance hints and provides .replace()"""
    def replace(self: T, **kwargs) -> T:
        # In a real scenario, this is overwritten by chex at runtime
        ...

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

type Stroke = Float[Array, "4"]
"""Float[Array, "4"], x_start, y_start, x_end, y_end; values in [0, 1]; (start,end) expected to be lexicographically ordered"""

type TargetStrokesCollection = Float[Array, "S 4"]
"""Float[Array, "S 4"], Each of the S strokes is (x_start, y_start, x_end, y_end); values in [0, 1]"""

type DrawnStrokesCollection = Float[Array, "T 4"]
"""Float[Array, "T 4"], T is max trial length, ie max strokes possible; each of the T strokes is (x_start, y_start, x_end, y_end); values in [0, 1]; (start,end) expected to be lexicographically ordered"""

type StrokesCollection = DrawnStrokesCollection | TargetStrokesCollection
"""Float[Array, "S|T 4"], (x_start, y_start, x_end, y_end) (start,end) expected to be lexicographically ordered, first dim shape depends whether those are targets or drawn"""

type TargetStrokesStatus = Bool[Array, "S"]
"""Bool[Array, "S"], for each target stroke if has been correctly covered"""

type TrialStep = int
"""Number of elapsed timesteps within one trial (not to be confused with BlockStep or TrainStep); values in {0..T}"""

type PolicyState = Optional[Float[Array, "H"] | EnvState]
"""Float[Array, "H"], Vector hidden state for the policy; can also be EnvState in the case of an OraclePolicy"""

type StepReward = Float
"""Reward obtained by the action; computed by update_stroke_status"""


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


@chex.dataclass(frozen=True)
class CanvasParams:
    num_target_strokes: int = 4
    max_num_strokes: int = 20
    size: int = 128
    stroke_min_length: float = 0.2
    stroke_max_length: float = 0.4
    # This ensures some gradient to help pinpoint center, but also visible
    thickness: float = 0.02
    softness: float = 0.02
    quality_max_pos_dif: float = .03  


@chex.dataclass(frozen=True)
class EnvState(JaxDataclass):
    '''
    Everything done to avoid carrying visual state between timesteps for reduced memory bandwidth
    NOTE: For now, keep rng state out of EnvState for more flexibility, might change later
    '''
    target_strokes: TargetStrokesCollection
    drawn_strokes: DrawnStrokesCollection
    position: CoordinateVector
    target_strokes_status: TargetStrokesStatus
    trial_step: TrialStep
    step_reward: StepReward

@chex.dataclass(frozen=True)
class StepCarry(JaxDataclass):
    policy_state: PolicyState
    env_state: EnvState

@chex.dataclass(frozen=True)
class Action(JaxDataclass):
    movement: MovementVector
    pressure: PressureValue

@chex.dataclass(frozen=True)
class StepOutput(JaxDataclass):
    env_state: EnvState
    action: Action
    obs: FullCanvas

class Policy(Protocol):
    def __call__(self, policy_step_rng_key: PolicyStepRngKey, canvas_params: CanvasParams, policy_state: PolicyState, observation: Optional[FullCanvas]) -> Tuple[PolicyState,Action]:
        ...
