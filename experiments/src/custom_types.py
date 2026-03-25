'''
Tons of boilerplate, but this makes nice usable tooltips in pylance so probably worth it

Try to only give types to things that might end up in a carried / returned pytree, not internals
'''
import chex 
from typing import Tuple, Protocol, TypeVar,dataclass_transform,Generic,Optional
from jaxtyping import Float, Array, Key, Bool, Int
from .config import EnvParams

# Strokes-related types:
type Stroke = Float[Array, "4"]
"""
Float[Array, "4"]\n
x_start, y_start, x_end, y_end; values in [0, 1]; (start,end) expected to be lexicographically ordered
"""

type TargetStrokes = Float[Array, "S 4"]
"""
Float[Array, "S 4"]\n
(x_start, y_start, x_end, y_end); values in [0, 1]; (start,end) expected to be lexicographically ordered
"""


type TargetStrokesBatch = Float[Array, "B S 4"]
"""
Float[Array, "B S 4"]\n
(x_start, y_start, x_end, y_end); values in [0, 1]; (start,end) expected to be lexicographically ordered
"""

type TargetStrokesHistory = Float[Array, "T B S 4"]
"""
Float[Array, "T B S 4"]\n
Time axis first since scan(vmap) is preferred\n
(x_start, y_start, x_end, y_end); values in [0, 1] (start,end) expected to be lexicographically ordered
"""

type DrawnStrokes = Float[Array, "T 4"]
"""
Float[Array, "T 4"], T is max trial length, ie max strokes possible\n
(x_start, y_start, x_end, y_end); values in [0, 1]; (start,end) expected to be lexicographically ordered
"""

type DrawnStrokesBatch = Float[Array, "B T 4"]
"""
Float[Array, "B T 4"], T is max trial length, ie max strokes possible\n
(x_start, y_start, x_end, y_end); values in [0, 1]; (start,end) expected to be lexicographically ordered
"""

type DrawnStrokesHistory = Float[Array, "T B T 4"]
"""
Float[Array, "T B T 4"], T is max trial length, ie max strokes possible\n
Time axis first since scan(vmap) is preferred\n
(x_start, y_start, x_end, y_end); values in [0, 1]; (start,end) expected to be lexicographically ordered
"""


# This keeps track of activ elines for reward computation
type TargetStrokesStatus = Bool[Array, "S"]
"""Bool[Array, "S"], for each target stroke if has been correctly covered"""

type TargetStrokesStatusBatch = Bool[Array, "B S"]
"""Bool[Array, "B S"], for each target stroke if has been correctly covered"""

type TargetStrokesStatusHistory = Bool[Array, "T B S"]
"""Bool[Array, "T B S"], for each target stroke if has been correctly covered"""


# Trial step
type TrialStep = Int[Array, ""]
"""Int[Array, "] the step in the current trial, not meant to be reused"""

type TrialStepBatch = Int[Array, "B"]
"""Int[Array, "B"] the step in the current trial, not meant to be reused"""

type TrialStepHistory = Int[Array, "T B"]
"""Int[Array, "T B"] the step in the current trial, not meant to be reused"""


# Policy states; force as arrays, can always just make it [0.]
type PolicyState = Float[Array, "H"]
"""Float[Array, H"] the current internal state of a policy"""

type PolicyStateBatch = Float[Array, "B H"]
"""Float[Array, "B H"] the current internal state of a policy"""

type PolicyStateHistory = Float[Array, "T B H"]
"""Float[Array, "T B H"] the current internal state of a policy"""


# Rewards
type Reward = Float[Array, ""]
"""Float[Array, ""] the reward obtained from stepping; scalar for now"""

type RewardBatch = Float[Array, "B"]
"""Float[Array, "B"] the reward obtained from stepping; scalar for now"""

type RewardHistory = Float[Array, "T B"]
"""Float[Array, "T B"] the reward obtained from stepping; scalar for now"""


# Heavy, not to be carried around mindlessly, image types
type SingleFrame = Float[Array, "H W"]
"""Float[Array, "H W"], Value range [0, 1[ for correct image conversion"""

type FullCanvas = Float[Array, "3 H W"]
"""Float[Array, "3 H W"], 3 is for RGB, aka pos, drawn, target; Value range [0, 1[ for correct image conversion"""

type CanvasBatch = Float[Array, "B 3 H W"]
"""Float[Array, "B 3 H W"], 3 is for RGB, aka pos, drawn, target; Value range [0, 1[ for correct image conversion"""

type CanvasHistory = Float[Array, "T B 3 H W"]
"""Float[Array, "T B 3 H W"], 3 is for RGB, aka pos, drawn, target; Value range [0, 1[ for correct image conversion"""


# Coordinates
type Coordinate = Float[Array, "2"]
"""Float[Array, "2"]: Value range [0, 1] to stay within frame"""

type CoordinateBatch = Float[Array, "B 2"]
"""Float[Array, "B 2"], Value range [0, 1] to stay within frame"""

type CoordinateHistory = Float[Array, "T B 2"]
"""Float[Array, "T B 2"], Value range [0, 1] to stay within frame"""

# Actions
type Action = Float[Array, "3"]
"""
Float[Array, "3"], Value range [-1, 1], movement_vector and pressure concatenated
IMPORTANT : drawing happens if Action[2] > 0, not 0.5, so tanh not sigmoid
"""

type ActionBatch = Float[Array, "B 3"]
"""
Float[Array, "B 3"], Value range [-1, 1], movement_vector and pressure concatenated
IMPORTANT : drawing happens if Action[2] > 0, not 0.5, so tanh not sigmoid
"""

type ActionHistory = Float[Array, "T B 3"]
"""
Float[Array, "T B 3"], Value range [-1, 1], movement_vector and pressure concatenated
IMPORTANT : drawing happens if Action[2] > 0, not 0.5, so tanh not sigmoid
"""

type KeyBatch = Key[Array, "B"]

Ty = TypeVar("Ty")

@dataclass_transform()
class JaxDataclass(Generic[Ty]):
    def replace(self, **kwargs) -> Ty:
        # Chex handles this at runtime
        ...

@chex.dataclass(frozen=True)
class EnvState(JaxDataclass):
    '''
    Everything done to avoid carrying visual state between timesteps for reduced memory bandwidth
    NOTE: For now, keep rng state out of EnvState for more flexibility, might change later

    target_strokes: Float[Array["S 4"]]
    drawn_strokes: Float[Array[ T 4"]]
    position: Float[Array["2"]]
    target_strokes_status: Bool[Array["S"]]
    trial_step: Int[Array[""]]
    '''
    target_strokes: TargetStrokes
    drawn_strokes: DrawnStrokes
    position: Coordinate
    target_strokes_status: TargetStrokesStatus
    trial_step: TrialStep
    
@chex.dataclass(frozen=True)
class EnvStateBatch(JaxDataclass):
    '''
    Everything done to avoid carrying visual state between timesteps for reduced memory bandwidth
    NOTE: For now, keep rng state out of EnvState for more flexibility, might change later

    target_strokes: Float[Array["B S 4"]]\n
    drawn_strokes: Float[Array[ B T 4"]]\n
    position: Float[Array["B 2"]]\n
    target_strokes_status: Bool[Array["B S"]]\n
    trial_step: Int[Array["B"]]\n
    '''
    target_strokes: TargetStrokesBatch
    drawn_strokes: DrawnStrokesBatch
    position: CoordinateBatch
    target_strokes_status: TargetStrokesStatusBatch
    trial_step: TrialStepBatch



@chex.dataclass(frozen=True)
class EnvStateHistory(JaxDataclass):
    '''
    Everything done to avoid carrying visual state between timesteps for reduced memory bandwidth
    NOTE: For now, keep rng state out of EnvState for more flexibility, might change later

    target_strokes: Float[Array["T B S 4"]]\n
    drawn_strokes: Float[Array["T B T 4"]]\n
    position: Float[Array["T B 2"]]\n
    target_strokes_status: Bool[Array["T B S"]]\n
    trial_step: Int[Array["T B"]]\n
    '''
    target_strokes: TargetStrokesHistory
    drawn_strokes: DrawnStrokesHistory
    position: CoordinateHistory
    target_strokes_status: TargetStrokesStatusHistory
    trial_step: TrialStepHistory

# NOTE: Since these are meant to be used by scan, assume they are already batched even if B=1 to encourage the 
# correct scan(vmap) pattern

@chex.dataclass(frozen=True)
class BatchedStepCarry(JaxDataclass):
    '''
    agent_state: PolicyStateBatch\n
    reference_state : PolicyStateBatch\n
    env_state: EnvStateBatch
    '''
    agent_state: PolicyStateBatch
    teacher_state : PolicyStateBatch
    env_state: EnvStateBatch

@chex.dataclass(frozen=True)
class RolloutStepOutput(JaxDataclass):
    '''
    NOTE: states and observations are pre-step

    env_state: EnvStateBatch\n
    obs: CanvasBatch
    agent_state: PolicyStateBatch\n
    teacher_state: PolicyStateBatch\n
    agent_action: ActionBatch\n
    teacher_action: ActionBatch\n
    agent_reward: RewardBatch
    teacher_reward: RewardBatch
    '''
    env_state: EnvStateBatch
    obs: CanvasBatch
    agent_state: PolicyStateBatch
    teacher_state: PolicyStateBatch
    agent_action: ActionBatch
    teacher_action: ActionBatch
    agent_reward: RewardBatch
    teacher_reward: RewardBatch


@chex.dataclass(frozen=True)
class FullRollout(JaxDataclass):
    '''
    NOTE: states and observations are pre-ste

    env_state: EnvStateHistory\n
    obs: CanvasHistory
    agent_state: PolicyStateHistory\n
    teacher_state: PolicyStateHistory\n
    agent_action: ActionHistory\n
    teacher_action: ActionHistory\n
    agent_reward: RewardHistory
    teacher_reward: RewardHistory
    '''
    env_state: EnvStateHistory
    obs: CanvasHistory
    agent_state: PolicyStateHistory
    teacher_state: PolicyStateHistory
    agent_action: ActionHistory
    teacher_action: ActionHistory
    agent_reward: RewardHistory
    teacher_reward: RewardHistory

class Policy(Protocol):
    '''
    Oracle policies also has access to underlying environment state on top of its own state and the observation.
    Agent policies should never use env_state !
    '''
    def __call__(self, rng_key: Key, policy_state: PolicyState, env_state: EnvState, observation: FullCanvas, env_params: EnvParams) -> Tuple[PolicyState,Action]:
        ...

class BatchedPolicy(Protocol):
    '''
    Oracle policies also has access to underlying environment state on top of its own state and the observation.
    Agent policies should never use env_state !
    '''
    def __call__(self, rng_key: Key[Array, "B"], policy_state: PolicyStateBatch, env_state: EnvStateBatch, observation: CanvasBatch, env_params: EnvParams) -> Tuple[PolicyStateBatch,ActionBatch]:
        ...


class PolicyStateInitializer(Protocol):
    def __call__(self, rng_key: Key[Array, ""]) -> PolicyState:
        ...

class PolicyStateBatchInitializer(Protocol):
    def __call__(self, rng_keys: Key[Array, "B"]) -> PolicyStateBatch:
        ...

class EnvStateInitializer(Protocol):
    def __call__(self, rng_key: Key[Array, ""], env_params: EnvParams) -> EnvState:
        ...

class EnvStateBatchInitializer(Protocol):
    def __call__(self, rng_keys: Key[Array, "B"], env_params: EnvParams) -> EnvStateBatch:
        ...