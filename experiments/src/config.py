import jax 
from typing import Literal

# This will be sued to sample the parameters for the rule of each trial
type RulesetLiteral = Literal[
    'along_cardinal_directions', # chooses between l2r, r2l, u2d, d2u
    'along_parametric_directions',
    'closest_to_center',
    'closest_to_random_anchor',
    'monotonic_line_length',
    'any_nonparametric',
    'any',
]

@jax.tree_util.register_static
class EnvParams:
    def __init__(self,
                num_target_strokes: int = 4,
                max_num_strokes: int = 10,
                size: int = 128,
                stroke_min_length: float = 0.1,
                stroke_max_length: float = 0.7,
                thickness: float = 0.01,
                softness: float = 0.03,
                line_done_cutoff: float = .05,
                false_draw_penalty: float = .1,
                ruleset: RulesetLiteral = 'along_cardinal_directions'
                ):
        self.num_target_strokes = num_target_strokes
        self.max_num_strokes=max_num_strokes
        self.size=size
        self.stroke_min_length=stroke_min_length
        self.stroke_max_length=stroke_max_length
        self.thickness=thickness
        self.softness=softness
        self.line_done_cutoff=line_done_cutoff    
        self.false_draw_penalty=false_draw_penalty    
        self.ruleset=ruleset
