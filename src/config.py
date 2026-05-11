import jax 
from typing import Literal, Optional, TypeAlias

# Repeat necessary here to have both static typing and the runtime checks
RULESET_OPTIONS = (
    'cardinal_directions', 
    'parametric_directions',
    'parametric_directions_with_decreasing',
    'modes_nonzero',
    'any',
    )

RulesetLiteral: TypeAlias = Literal[
    'cardinal_directions', 
    'parametric_directions',
    'parametric_directions_with_decreasing',
    'modes_nonzero',
    'any',
    ]

@jax.tree_util.register_static
class EnvParams:
    def __init__(self,
                num_target_strokes: int = 5,
                size: int = 128,
                stroke_min_length: float = 0.1,
                stroke_max_length: float = 0.7,
                thickness: float = 0.01,
                softness: float = 0.03,
                line_done_cutoff: float = .05,
                false_draw_penalty: float = .1,
                ruleset: RulesetLiteral = 'cardinal_directions',
                max_num_strokes: Optional[int] = None, 
                ):
        self.num_target_strokes = num_target_strokes
        self.max_num_strokes=2*num_target_strokes if max_num_strokes is None else max_num_strokes 
        self.size=size
        self.stroke_min_length=stroke_min_length
        self.stroke_max_length=stroke_max_length
        self.thickness=thickness
        self.softness=softness
        self.line_done_cutoff=line_done_cutoff    
        self.false_draw_penalty=false_draw_penalty    
        self.ruleset=ruleset
