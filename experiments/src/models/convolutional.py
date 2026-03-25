import jax
import jax.numpy as jnp
import equinox as eqx
import chex
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class ConvStageConfig:
    num_blocks: int
    out_channels: int
    stride: int # Stride for the *first* block in the stage

@dataclass(frozen=True)
class MultiStageConfig:
    in_channels: int
    stages: tuple[ConvStageConfig, ...] 

class CoordConv2d(eqx.nn.Conv2d):
    """
    Subclass of eqx.nn.Conv2d that prepends normalized (x, y) coordinates to the input channels.
    Apparently can help in visual regression tasks (avoids having to embed coordinate map in weights,
    saving more training gradient and representation power for more interesting stuff)
    """

    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs):
        # Need to add 2 to in_channels for coordinate maps, rest kept identical
        super().__init__(in_channels + 2, out_channels, *args, **kwargs)

    def __call__(self, x):
        C, H, W = x.shape
        y_coords = jnp.linspace(-1, 1, H)
        x_coords = jnp.linspace(-1, 1, W)
        y_grid, x_grid = jnp.meshgrid(y_coords, x_coords, indexing='ij')
        y_grid = jnp.expand_dims(y_grid, axis=0)
        x_grid = jnp.expand_dims(x_grid, axis=0)
        x_augmented = jnp.concatenate([x, y_grid, x_grid], axis=0)
        return super().__call__(x_augmented)


class ResBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    shortcut: eqx.Module | None

    def __init__(self, in_channels: int, out_channels: int, stride: int, key: jax.Array, norm_groups: int=32, use_coord_conv: bool=False):
        k1, k2, k3 = jax.random.split(key, 3)
        
        # conv1 handles the spatial downsampling if stride > 1
        if use_coord_conv:
            self.conv1 = CoordConv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, key=k1
            )
        else:
            self.conv1 = eqx.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, key=k1
            )
        self.conv2 = eqx.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, key=k2
        )
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = eqx.nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, use_bias=False, key=k3
            )
        else:
            self.shortcut = None 

        self.norm1 = eqx.nn.GroupNorm(groups=norm_groups, channels=out_channels)
        self.norm2 = eqx.nn.GroupNorm(groups=norm_groups, channels=out_channels)

    def __call__(self, x: chex.Array) -> chex.Array:
        h = jax.nn.relu(self.conv1(x))
        h = self.norm1(h)
        h = self.conv2(h)
        h = self.norm2(h)
        residual = x if self.shortcut is None else self.shortcut(x)
        return jax.nn.relu(residual + h)

class MultiStageConvnet(eqx.Module):
    blocks: tuple[ResBlock, ...]
    out_channels: int

    def __init__(self, config: MultiStageConfig, key: jax.Array):
        total_blocks = sum(stage.num_blocks for stage in config.stages)
        keys = jax.random.split(key, total_blocks + 1)
        
        blocks_list = []
        current_channels = config.in_channels
        key_idx = 0
        
        for stage in config.stages:
            for block_idx in range(stage.num_blocks):
                # Apply the stride/CoordConv only on the first block of the stage
                stride = stage.stride if block_idx == 0 else 1
                coord_conv = (block_idx == 0) 
                out_channels = stage.out_channels
                
                blocks_list.append(
                    ResBlock(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        stride=stride,
                        key=keys[key_idx],
                        use_coord_conv=coord_conv,
                    )
                )
                
                current_channels = out_channels
                key_idx += 1
                
        self.blocks = tuple(blocks_list)    
        self.out_channels = config.stages[-1].out_channels   

    def __call__(self, x: chex.Array) -> chex.Array:
        for block in self.blocks:
            x = block(x)
        return x




small_model_config = MultiStageConfig(
    in_channels=3,
    stages=(
        ConvStageConfig(num_blocks=1, out_channels=32, stride=2), 
        ConvStageConfig(num_blocks=2, out_channels=64, stride=2), 
        ConvStageConfig(num_blocks=2, out_channels=128, stride=2), 
        ConvStageConfig(num_blocks=2, out_channels=256, stride=2), 
    ),
)
medium_model_config = MultiStageConfig(
    in_channels=3,
    stages=(
        ConvStageConfig(num_blocks=2, out_channels=64, stride=2), 
        ConvStageConfig(num_blocks=2, out_channels=128, stride=2), 
        ConvStageConfig(num_blocks=1, out_channels=256, stride=2), 
        ConvStageConfig(num_blocks=1, out_channels=512, stride=2), 
    ),
)

convnet_config_register = {
    "small": small_model_config, 
    "medium": medium_model_config,
                           }

def build_convnet(config: Optional[MultiStageConfig]=None, config_name:Optional[str]=None, seed: int = 777) -> MultiStageConvnet:
    if config is None:
        assert config_name is not None
        config = convnet_config_register[config_name]
    key = jax.random.key(seed)
    return MultiStageConvnet(config, key)
