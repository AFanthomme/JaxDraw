import jax 
import chex

@jax.tree_util.register_static
class EnvParams:
    def __init__(self,
                num_target_strokes: int = 4,
                max_num_strokes: int = 10,
                size: int = 128,
                stroke_min_length: float = 0.1,
                stroke_max_length: float = 0.7,
                thickness: float = 0.02,
                softness: float = 0.03,
                quality_max_pos_dif: float = .03 ,
                ):
        self.num_target_strokes = num_target_strokes
        self.max_num_strokes=max_num_strokes
        self.size=size
        self.stroke_min_length=stroke_min_length
        self.stroke_max_length=stroke_max_length
        self.thickness=thickness
        self.softness=softness
        self.quality_max_pos_dif=quality_max_pos_dif    

    @classmethod
    def debug_config(cls) -> EnvParams:
        """Returns a version of params optimized for fast debugging."""
        return cls(
            num_target_strokes=2,
            max_num_strokes=5,
        )
    
    def __setattr__(self, attr, value):
        raise AttributeError("Trying to set attribute on a frozen instance")