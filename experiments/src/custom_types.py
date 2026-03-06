'''
Tons of boilerplate, but this makes nice usable tooltips in pylance so probably worth it

Try to only give types to things that might end up in a carried / returned pytree, not internals
'''
import chex 
from typing import Tuple, Protocol, TypeVar,dataclass_transform,Generic
from jaxtyping import Float, Array, Key, Bool, Int
from config import EnvParams

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


type BatchedTargetStrokes = Float[Array, "B S 4"]
"""
Float[Array, "B S 4"]\n
(x_start, y_start, x_end, y_end); values in [0, 1]; (start,end) expected to be lexicographically ordered
"""

type HistoryTargetStrokes = Float[Array, "T B S 4"]
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

type BatchedDrawnStrokes = Float[Array, "B T 4"]
"""
Float[Array, "B T 4"], T is max trial length, ie max strokes possible\n
(x_start, y_start, x_end, y_end); values in [0, 1]; (start,end) expected to be lexicographically ordered
"""

type HistoryDrawnStrokes = Float[Array, "T B T 4"]
"""
Float[Array, "T B T 4"], T is max trial length, ie max strokes possible\n
Time axis first since scan(vmap) is preferred\n
(x_start, y_start, x_end, y_end); values in [0, 1]; (start,end) expected to be lexicographically ordered
"""


# This keeps track of activ elines for reward computation
type TargetStrokesStatus = Bool[Array, "S"]
"""Bool[Array, "S"], for each target stroke if has been correctly covered"""

type BatchedTargetStrokesStatus = Bool[Array, "B S"]
"""Bool[Array, "B S"], for each target stroke if has been correctly covered"""

type HistoryTargetStrokesStatus = Bool[Array, "T B S"]
"""Bool[Array, "T B S"], for each target stroke if has been correctly covered"""


# Trial step
type TrialStep = Int[Array, ""]
"""Int[Array, "] the step in the current trial, not meant to be reused"""

type BatchedTrialStep = Int[Array, "B"]
"""Int[Array, "B"] the step in the current trial, not meant to be reused"""

type HistoryTrialStep = Int[Array, "T B"]
"""Int[Array, "T B"] the step in the current trial, not meant to be reused"""


# Policy states; force as arrays, can always just make it [0.]
type PolicyState = Float[Array, "H"]
"""Float[Array, H"] the current internal state of a policy"""

type BatchedPolicyState = Float[Array, "B H"]
"""Float[Array, "B H"] the current internal state of a policy"""

type HistoryPolicyState = Float[Array, "T B H"]
"""Float[Array, "T B H"] the current internal state of a policy"""


# Rewards
type StepReward = Float[Array, ""]
"""Float[Array, ""] the reward obtained from stepping; scalar for now"""

type BatchedStepReward = Float[Array, "B"]
"""Float[Array, "B"] the reward obtained from stepping; scalar for now"""

type HistoryStepReward = Float[Array, "T B"]
"""Float[Array, "T B"] the reward obtained from stepping; scalar for now"""


# Heavy, not to be carried around mindlessly, image types
type SingleFrame = Float[Array, "H W"]
"""Float[Array, "H W"], Value range [0, 1[ for correct image conversion"""

type FullCanvas = Float[Array, "3 H W"]
"""Float[Array, "3 H W"], 3 is for RGB, aka pos, drawn, target; Value range [0, 1[ for correct image conversion"""

type BatchedCanvas = Float[Array, "B 3 H W"]
"""Float[Array, "B 3 H W"], 3 is for RGB, aka pos, drawn, target; Value range [0, 1[ for correct image conversion"""

type HistoryCanvas = Float[Array, "T B 3 H W"]
"""Float[Array, "T B 3 H W"], 3 is for RGB, aka pos, drawn, target; Value range [0, 1[ for correct image conversion"""


# Coordinates
type Coordinate = Float[Array, "2"]
"""Float[Array, "2"]: Value range [0, 1] to stay within frame"""

type BatchedCoordinate = Float[Array, "B 2"]
"""Float[Array, "B 2"], Value range [0, 1] to stay within frame"""

type HistoryCoordinate = Float[Array, "T B 2"]
"""Float[Array, "T B 2"], Value range [0, 1] to stay within frame"""

# Actions
type Action = Float[Array, "3"]
"""
Float[Array, "3"], Value range [-1, 1], movement_vector and pressure concatenated
IMPORTANT : drawing happens if Action[2] > 0, not 0.5, so tanh not sigmoid
"""

type BatchedAction = Float[Array, "B 3"]
"""
Float[Array, "B 3"], Value range [-1, 1], movement_vector and pressure concatenated
IMPORTANT : drawing happens if Action[2] > 0, not 0.5, so tanh not sigmoid
"""

type HistoryAction = Float[Array, "T B 3"]
"""
Float[Array, "T B 3"], Value range [-1, 1], movement_vector and pressure concatenated
IMPORTANT : drawing happens if Action[2] > 0, not 0.5, so tanh not sigmoid
"""

T = TypeVar("T")

@dataclass_transform()
class JaxDataclass(Generic[T]):
    def replace(self, **kwargs) -> T:
        # Chex handles this at runtime
        ...

@chex.dataclass(frozen=True)
class HistoryEnvState(JaxDataclass):
    '''
    Everything done to avoid carrying visual state between timesteps for reduced memory bandwidth
    NOTE: For now, keep rng state out of EnvState for more flexibility, might change later

    target_strokes: Float[Array["T B S 4"]]\n
    drawn_strokes: Float[Array["T B T 4"]]\n
    position: Float[Array["T B 2"]]\n
    target_strokes_status: Bool[Array["T B S"]]\n
    trial_step: Int[Array["T B"]]\n
    step_reward: Float[Array["T B"]]
    '''
    target_strokes: HistoryTargetStrokes
    drawn_strokes: HistoryDrawnStrokes
    position: HistoryCoordinate
    target_strokes_status: HistoryTargetStrokesStatus
    trial_step: HistoryTrialStep
    step_reward: HistoryStepReward

@chex.dataclass(frozen=True)
class BatchedEnvState(JaxDataclass):
    '''
    Everything done to avoid carrying visual state between timesteps for reduced memory bandwidth
    NOTE: For now, keep rng state out of EnvState for more flexibility, might change later

    target_strokes: Float[Array["B S 4"]]\n
    drawn_strokes: Float[Array[ B T 4"]]\n
    position: Float[Array["B 2"]]\n
    target_strokes_status: Bool[Array["B S"]]\n
    trial_step: Int[Array["B"]]\n
    step_reward: Float[Array["B"]]
    '''
    target_strokes: BatchedTargetStrokes
    drawn_strokes: BatchedDrawnStrokes
    position: BatchedCoordinate
    target_strokes_status: BatchedTargetStrokesStatus
    trial_step: BatchedTrialStep
    step_reward: BatchedStepReward

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
    step_reward: Float[Array[""]]
    '''
    target_strokes: TargetStrokes
    drawn_strokes: DrawnStrokes
    position: Coordinate
    target_strokes_status: TargetStrokesStatus
    trial_step: TrialStep
    step_reward: StepReward


# NOTE: Since these are meant to be used by scan, assume they are already batched even if B=1 to encourage the 
# correct scan(vmap) pattern

@chex.dataclass(frozen=True)
class BatchedStepCarry(JaxDataclass):
    '''
    agent_state: BatchedPolicyState\n
    reference_state : BatchedPolicyState\n
    env_state: BatchedEnvState
    '''
    agent_state: BatchedPolicyState
    teacher_state : BatchedPolicyState
    env_state: BatchedEnvState

@chex.dataclass(frozen=True)
class BatchedStepOutput(JaxDataclass):
    '''
    env_state: BatchedEnvState\n
    agent_state: BatchedPolicyState\n
    reference_state: BatchedPolicyState\n
    agent_action: BatchedAction\n
    reference_action: BatchedAction\n
    obs: BatchedCanvas
    '''
    env_state: BatchedEnvState
    agent_state: BatchedPolicyState
    teacher_state: BatchedPolicyState
    agent_action: BatchedAction
    teacher_action: BatchedAction
    obs: BatchedCanvas


@chex.dataclass(frozen=True)
class HistoryOutput(JaxDataclass):
    '''
    env_state: HistoryEnvState\n
    agent_state: HistoryPolicyState\n
    reference_state: HistoryPolicyState\n
    agent_action: HistoryAction\n
    reference_action: HistoryAction\n
    obs: HistoryCanvas
    '''
    env_state: HistoryEnvState
    agent_state: HistoryPolicyState
    teacher_state: HistoryPolicyState
    agent_action: HistoryAction
    teacher_action: HistoryAction
    obs: HistoryCanvas


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
    def __call__(self, rng_key: Key[Array, "B"], policy_state: BatchedPolicyState, env_state: BatchedEnvState, observation: BatchedCanvas, env_params: EnvParams) -> Tuple[BatchedPolicyState,BatchedAction]:
        ...


class PolicyStateInitializer(Protocol):
    def __call__(self, rng_key: Key[Array, ""]) -> PolicyState:
        ...

class BatchedPolicyStateInitializer(Protocol):
    def __call__(self, rng_keys: Key[Array, "B"]) -> BatchedPolicyState:
        ...

class EnvStateInitializer(Protocol):
    def __call__(self, rng_key: Key[Array, ""], env_params: EnvParams) -> EnvState:
        ...

class BatchedEnvStateInitializer(Protocol):
    def __call__(self, rng_keys: Key[Array, "B"], env_params: EnvParams) -> BatchedEnvState:
        ...