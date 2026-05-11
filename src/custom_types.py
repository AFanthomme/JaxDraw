import chex 
from typing import Tuple, Protocol, TypeVar, dataclass_transform, Generic
from jaxtyping import Float, Array, Key, Bool, Int
from .config import EnvParams, RulesetLiteral
import jax
import dataclasses

type Stroke = Float[Array, "4"]
"Float[4] : x_start, y_start, x_end, y_end; values in [0, 1]; (start,end) expected to be lexicographically ordered"

type TargetStrokes = Float[Array, "S 4"]
"Float[S 4] : (x_start, y_start, x_end, y_end); values in [0, 1]; (start,end) expected to be lexicographically ordered"

type TargetStrokesBatch = Float[Array, "B S 4"]
"Float[B S 4] : (x_start, y_start, x_end, y_end); values in [0, 1]; (start,end) expected to be lexicographically ordered"

type TargetStrokesSequence = Float[Array, "B S 4"]
"Float[B S 4] : (x_start, y_start, x_end, y_end); values in [0, 1]; (start,end) expected to be lexicographically ordered"

type TargetStrokesHistory = Float[Array, "T B S 4"]
"Float[T B S 4] : (x_start, y_start, x_end, y_end); values in [0, 1] (start,end) expected to be lexicographically ordered"

type DrawnStrokes = Float[Array, "T 4"]
"Float[T 4],(x_start, y_start, x_end, y_end); values in [0, 1] (start,end) expected to be lexicographically ordered"

type DrawnStrokesBatch = Float[Array, "B T 4"]
"Float[B T 4], (x_start, y_start, x_end, y_end); values in [0, 1] (start,end) expected to be lexicographically ordered"

type DrawnStrokesSequence = Float[Array, "B T 4"]
"Float[T T 4], (x_start, y_start, x_end, y_end); values in [0, 1] (start,end) expected to be lexicographically ordered"

type DrawnStrokesHistory = Float[Array, "T B T 4"]
"Float[T B T 4], (x_start, y_start, x_end, y_end); values in [0, 1] (start,end) expected to be lexicographically ordered"

type TargetStrokesStatus = Bool[Array, "S"]
"Bool[S], for each target stroke if has been correctly covered"

type TargetStrokesStatusSequence = Bool[Array, "B S"]
"Bool[B S], for each target stroke if has been correctly covered"

type TargetStrokesStatusBatch = Bool[Array, "B S"]
"Bool[B S], for each target stroke if has been correctly covered"

type TargetStrokesStatusHistory = Bool[Array, "T B S"]
"Bool[T B S], for each target stroke if has been correctly covered"

type TrialStep = Int[Array, ""]
"Int : step in the current trial"

type TrialStepBatch = Int[Array, "B"]
"Int[B] : step in the current trial"

type TrialStepSequence = Int[Array, "B"]
"Int[B] : step in the current trial"

type TrialStepHistory = Int[Array, "T B"]
"Int[T B] : step in the current trial"

type PolicyState = Float[Array, "H"]
"Float[H] : the current internal state of policy"

type PolicyStateBatch = Float[Array, "B H"]
"Float[B H] : the current internal state of policy"

type PolicyStateSequence = Float[Array, "B H"]
"Float[B H] : the current internal state of policy"

type PolicyStateHistory = Float[Array, "T B H"]
"Float[T B H] : the current internal state of policy" 

type Reward = Float[Array, ""]
"Float the reward obtained from stepping; scalar for now"

type RewardBatch = Float[Array, "B"]
"Float[B] the reward obtained from stepping; scalar for now"

type RewardSequence = Float[Array, "T"]
"Float[T] the reward obtained from stepping; scalar for now"

type RewardHistory = Float[Array, "T B"]
"Float[T B] the reward obtained from stepping; scalar for now"

type SingleFrame = Float[Array, "H W"]
"Float[H W], Value range [0, 1[ for correct image conversion"

type FullCanvas = Float[Array, "3 H W"]
"Float[3 H W], 3 is for RGB, aka pos, drawn, target; Value range [0, 1[ for correct image conversion"

type CanvasBatch = Float[Array, "B 3 H W"]
"Float[B 3 H W], 3 is for RGB, aka pos, drawn, target; Value range [0, 1[ for correct image conversion"

type CanvasSequence = Float[Array, "T 3 H W"]
"Float[T 3 H W], no batch but time dim present, 3 is for RGB, aka pos, drawn, target; Value range [0, 1[ for correct image conversion"

type CanvasHistory = Float[Array, "T B 3 H W"]
"Float[T B 3 H W], 3 is for RGB, aka pos, drawn, target; Value range [0, 1[ for correct image conversion"

type Coordinate = Float[Array, "2"]
"Float[2]: Value range [0, 1] to stay within frame"

type CoordinateBatch = Float[Array, "B 2"]
"Float[B 2], Value range [0, 1] to stay within frame"

type CoordinateSequence = Float[Array, "B 2"]
"Float[T 2], Value range [0, 1] to stay within frame"

type CoordinateHistory = Float[Array, "T B 2"]
"Float[T B 2], Value range [0, 1] to stay within frame"

type Action = Float[Array, "3"]
"""
Float[3], Value range [-1, 1], movement_vector and pressure concatenated\n
Drawing happens if Action[2] > 0, not 0.5
"""

type ActionBatch = Float[Array, "B 3"]
"""
Float[B 3], Value range [-1, 1], movement_vector and pressure concatenated\n
Drawing happens if Action[2] > 0, not 0.5
"""

type ActionSequence = Float[Array, "T 3"]
"""
Float[T 3], Value range [-1, 1], movement_vector and pressure concatenated\n
Drawing happens if Action[2] > 0, not 0.5
"""

type ActionHistory = Float[Array, "T B 3"]
"""
Float[T B 3], Value range [-1, 1], movement_vector and pressure concatenated\n
Drawing happens if Action[2] > 0, not 0.5
"""

type SortMode = Int[Array, ""]
"Int : 0=projection, 1=proximity, 2=length, 3=ego_spiral, 4=allo_spiral"

type SortModeBatch = Int[Array, "B"]
"Int[B] : 0=projection, 1=proximity, 2=length, 3=ego_spiral, 4=allo_spiral"

type SortModeSequence = Int[Array, "T"]
"Int[T] : 0=projection, 1=proximity, 2=length, 3=ego_spiral, 4=allo_spiral"

type SortModeHistory = Int[Array, "T B"]
"Int[T B] : 0=projection, 1=proximity, 2=length, 3=ego_spiral, 4=allo_spiral"

type RefAngle = Float[Array, ""]
"Float[T B] : Angle (radians) used in the line ordering"

type RefAngleBatch = Float[Array, "B"]
"Float[B] : Angle (radians) used in the line ordering"

type RefAngleSequence = Float[Array, "T"]
"Float[T] : Angle (radians) used in the line ordering"

type RefAngleHistory = Float[Array, "T B"]
"Float[T B] : Angle (radians) used in the line ordering"

type RefPoint = Float[Array, "2"]
"Float[2] : Reference point used in the line ordering"

type RefPointBatch = Float[Array, "B 2"]
"Float[B 2] : Reference point used in the line ordering"

type RefPointSequence = Float[Array, "T 2"]
"Float[T 2] : Reference point used in the line ordering"

type RefPointHistory = Float[Array, "T B 2"]
"Float[T B 2] : Reference point used in the line ordering"

type Decreasing = Bool[Array, ""]
"Bool : reverse the line ordering"

type DecreasingBatch = Bool[Array, "B"]
"Bool[B] : reverse the line ordering"

type DecreasingSequence = Bool[Array, "T"]
"Bool[T] : reverse the line ordering"

type DecreasingHistory = Bool[Array, "T B"]
"Bool[T B] : reverse the line ordering"

type StillFollowingOrdering = Bool[Array, ""]
"Bool : track if drawing order still respected until current step"

type StillFollowingOrderingBatch = Bool[Array, "B"]
"Bool[B] : track if drawing order still respected until current step"

type StillFollowingOrderingSequence = Bool[Array, "T"]
"Bool[T] : track if drawing order still respected until current step"

type StillFollowingOrderingHistory = Bool[Array, "T B"]
"Bool[T B] : track if drawing order still respected until current step"

type KeyBatch = Key[Array, "B"]

Ty = TypeVar("Ty")
@dataclass_transform()
class JaxDataclass(Generic[Ty]):
    def replace(self, **kwargs) -> Ty:
        ...

    def __getitem__(self, idx):
            """
            Applies the slicing index across all leaves in the pytree.
            For example, obj[:, b] passes idx=(slice(None), b) to every array.
            """
            return jax.tree_util.tree_map(lambda x: x[idx], self)

    def as_type(self, target_class: type):
            """
            Casts the current dataclass to a new dataclass type, 
            assuming identical field names.
            """
            # Extract the fields from the current instance as a dictionary
            field_dict = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
            # Instantiate the target class with the sliced fields
            return target_class(**field_dict)

@chex.dataclass(frozen=True)
class EnvState(JaxDataclass):
    '''
    target_strokes: Float[S 4]\n
    drawn_strokes: Float[T 4]\n
    position: Float[2]\n
    target_strokes_status: Bool[S]\n
    trial_step: Int\n
    sort_mode: Int\n
    ref_angle: Float\n
    ref_point: Float[2]\n
    decreasing: Bool\n
    still_following_ordering:  Bool
    '''
    target_strokes: TargetStrokes
    drawn_strokes: DrawnStrokes
    position: Coordinate
    target_strokes_status: TargetStrokesStatus
    trial_step: TrialStep
    sort_mode: SortMode
    ref_angle: RefAngle
    ref_point: RefPoint
    decreasing: Decreasing
    still_following_ordering: StillFollowingOrdering
    
@chex.dataclass(frozen=True)
class EnvStateBatch(JaxDataclass):
    '''
    target_strokes: Float[B S 4]\n
    drawn_strokes: Float[B T 4]\n
    position: Float[B 2]\n
    target_strokes_status: Bool[B S]\n
    trial_step: Int[B]\n
    sort_mode: Int[B]\n
    ref_angle: Float[B]\n
    ref_point: Float[B 2]\n
    decreasing: Bool[B]\n
    still_following_ordering:  Bool[B]
    '''
    target_strokes: TargetStrokesBatch
    drawn_strokes: DrawnStrokesBatch
    position: CoordinateBatch
    target_strokes_status: TargetStrokesStatusBatch
    trial_step: TrialStepBatch
    sort_mode: SortModeBatch
    ref_angle: RefAngleBatch
    ref_point: RefPointBatch
    decreasing: DecreasingBatch
    still_following_ordering: StillFollowingOrderingBatch

@chex.dataclass(frozen=True)
class EnvStateSequence(JaxDataclass):
    '''
    target_strokes: Float[T S 4]\n
    drawn_strokes: Float[T T 4]\n
    position: Float[T 2]\n
    target_strokes_status: Bool[T S]\n
    trial_step: Int[T]\n
    sort_mode: Int[T]\n
    ref_angle: Float[T]\n
    ref_point: Float[T 2]\n
    decreasing: Bool[T]\n
    still_following_ordering:  Bool[T]
    '''
    target_strokes: TargetStrokesSequence
    drawn_strokes: DrawnStrokesSequence
    position: CoordinateSequence
    target_strokes_status: TargetStrokesStatusSequence
    trial_step: TrialStepSequence
    sort_mode: SortModeSequence
    ref_angle: RefAngleSequence
    ref_point: RefPointSequence
    decreasing: DecreasingSequence
    still_following_ordering: StillFollowingOrderingSequence

@chex.dataclass(frozen=True)
class EnvStateHistory(JaxDataclass):
    '''
    target_strokes: Float[T B S 4]\n
    drawn_strokes: Float[T B T 4]\n
    position: Float[T B 2]\n
    target_strokes_status: Bool[T B S]\n
    trial_step: Int[T B]\n
    sort_mode: Int[T B]\n
    ref_angle: Float[T B]\n
    ref_point: Float[T B 2]\n
    decreasing: Bool[T B]\n
    still_following_ordering:  Bool[T B]
    '''
    target_strokes: TargetStrokesHistory
    drawn_strokes: DrawnStrokesHistory
    position: CoordinateHistory
    target_strokes_status: TargetStrokesStatusHistory
    trial_step: TrialStepHistory
    sort_mode: SortModeHistory
    ref_angle: RefAngleHistory
    ref_point: RefPointHistory
    decreasing: DecreasingHistory
    still_following_ordering: StillFollowingOrderingHistory


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
    NOTE: states and observations are pre-step\n
    env_state: EnvStateBatch\n
    obs: CanvasBatch
    agent_state: PolicyStateBatch\n
    teacher_state: PolicyStateBatch\n
    agent_action: ActionBatch\n
    teacher_action: ActionBatch\n
    agent_reward: RewardBatch\n
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
    NOTE: states and observations are pre-step\n
    env_state: EnvStateHistory\n
    obs: CanvasHistory
    agent_state: PolicyStateHistory\n
    teacher_state: PolicyStateHistory\n
    agent_action: ActionHistory\n
    teacher_action: ActionHistory\n
    agent_reward: RewardHistory\n
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
    def __call__(self, rng_key: Key, policy_state: PolicyState, env_state: EnvState, observation: FullCanvas, env_params: EnvParams) -> Tuple[PolicyState,Action]:
        ...

class BatchedPolicy(Protocol):
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