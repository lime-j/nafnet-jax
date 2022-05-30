# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Sequence, TypeVar
import functools
import einops
from flax import linen as nn
import jax.numpy as jnp
import jax
T = TypeVar('T')

class GlobalAvgPool1D(nn.Module):
    @nn.compact
    def __call__(self, x):
        return jnp.mean(x, axis=-1, keepdims=True)

class GlobalAvgPool2D(nn.Module):
    @nn.compact
    def __call__(self, x):
        return jnp.mean(x, axis=(-3, -2), keepdims=True)


class SimpleGate(nn.Module):
    @nn.compact
    def __call__(self, x) :
        x1, x2 = x.split(2, axis=-1)
        return x1 * x2

class PixelShuffle(nn.Module):
    scale_factor: int

    def setup(self):
        self.layer = functools.partial(
            einops.rearrange,
            pattern="b h w (h2 w2 c) -> b (h h2) (w w2) c",
            h2=self.scale_factor,
            w2=self.scale_factor
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.layer(x)

class SimpleAttention(nn.Module):
    out_channels : int
    @nn.compact
    def __call__(self, x) :
        x = GlobalAvgPool2D()(x)
        x = nn.Dense(features=self.out_channels)(x)
        return x
        
class NAFBlock(nn.Module):
    c : int
    dw_expand : int = 2
    ffn_expand : int = 2
    drop_out_rate : float = 0.
    
    def setup(self):
        self.dw_channel = self.c * self.dw_expand
        self.ffn_channel = self.c * self.ffn_expand
        self.gamma = self.param('gamma', lambda key, shape, dtype: jnp.zeros(shape, dtype), (1, 1, 1, self.c), jnp.float32)
        self.beta = self.param('beta', lambda key, shape, dtype: jnp.zeros(shape, dtype), (1, 1, 1, self.c), jnp.float32)
    @nn.compact
    def __call__(self, ipt) :
        
        x = nn.LayerNorm()(ipt)
        x = nn.Dense(features=self.dw_channel, name="conv1")(x)
        x = nn.Conv(features=self.dw_channel, kernel_size=(3, 3), 
                    padding='SAME', use_bias = True,
                    feature_group_count=self.dw_channel, name="conv2")(x)
        x = SimpleGate()(x)
        scale = SimpleAttention(self.dw_channel // 2)(x)
        x = x * scale
        x = nn.Conv(self.c, kernel_size=(1, 1),
                    strides=(1, 1), use_bias=True, name='conv3')(x)   
        y = ipt + x * self.beta
        x = nn.LayerNorm()(y)
        x = nn.Dense(features=self.ffn_channel, name="conv4")(x)
        x = SimpleGate()(x)
        x = nn.Dense(features=self.c, name="conv5")(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    enc_blk_nums : list
    dec_blk_nums : list
    img_channel : int = 3
    width : int = 16 #channels
    middle_blk_num : int = 1
    name : str = None


    def setup(self):
        self.padder_size = 2 ** len(self.enc_blk_nums) 
    @nn.compact
    def __call__(self, inputs, train=True) :
        B, H, W, C = inputs.shape
        mod_pad_h = (self.padder_size - H % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - W % self.padder_size) % self.padder_size 
        inputs = jnp.pad(inputs, ((0, 0), (0, mod_pad_h), (0, mod_pad_w), (0, 0)), mode='constant')
        x = nn.Conv(self.width, kernel_size=(3, 3), use_bias=True, name='intro')(inputs)
        encs = []
        chan = self.width
        for blk_nums in self.enc_blk_nums :
            x = nn.Sequential([NAFBlock(chan) for _ in range(blk_nums)])(x)
            encs.append(x)
            x = nn.Conv(chan * 2, kernel_size=(2, 2), strides=(2, 2), padding=(0, 0))(x)
            chan = chan * 2
        x = nn.Sequential([NAFBlock(chan) for _ in range(self.middle_blk_num)])(x)
        for blk_nums, enc_skip in zip(self.dec_blk_nums, encs[::-1]):
            x = nn.Sequential([nn.Conv(chan * 2, kernel_size=(1, 1), use_bias=False),
                               PixelShuffle(2)])(x)
            x = x + enc_skip
            chan = chan // 2
            x = nn.Sequential([NAFBlock(chan) for _ in range(blk_nums)])(x)
        x = inputs + nn.Conv(self.img_channel, kernel_size=(3, 3), use_bias=True, name='ending')(x)
        return jax.lax.dynamic_slice(x, (0, 0, 0, 0), (B, H, W, C))
