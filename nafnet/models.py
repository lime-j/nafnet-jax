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

from flax import linen as nn
import jax.numpy as jnp

T = TypeVar('T')

class GlobalAvgPool1D(nn.Module):
    @nn.compact
    def __call__(self, x):
        return jnp.mean(x, axis=-1)

class GlobalAvgPool2D(nn.Module):
    @nn.compact
    def __call__(self, x):
        return jnp.mean(x, axis=(-3, -2))


class SimpleGate(nn.Module):
    @nn.compact
    def __call__(self, x) :
        x1, x2 = x.split(2, axis=-1)
        return x1 * x2

class PixelShuffle(nn.Module):
    scale_factor: int

    def setup(self):
        self.layer = partial(
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
        x = nn.Conv(self.out_channels, kernel_size=(1, 1), 
                    strides=(1, 1), use_bias=True, 
                    padding="SAME")(x)
        return x
        
class NAFBlock(nn.Module):
    c : int
    dw_expand : int = 2
    ffn_expand : int = 2
    drop_out_rate : float = 0.
    
    def setup(self):
        self.dw_channel = c * dw_expand
        self.ffn_channel = c * ffn_expand
        self.gamma = self.param('gamma', lambda key, shape, dtype: jnp.zeros(shape, dtype), (1, 1, 1, c), jnp.float32)
        self.beta = self.param('beta', lambda key, shape, dtype: jnp.zeros(shape, dtype), (1, 1, 1, c), jnp.float32)
    @nn.compact
    def __call__(self, ipt) :
        
        x = nn.LayerNorm()(ipt)
        x = nn.Dense(features=self.dw_channel, name="conv1")(x)
        x = nn.Conv(features=self.dw_channel, kernel_size=(3, 3), 
                    padding='SAME', use_bias = True,
                    feature_group_count=self.dw_channel, name="conv2")(x)
        x = SimpleGate()(x)
        x = x * SimpleAttention(self.dw_channel // 2)(x)
        x = nn.Conv(self.dw_channel, kernel_size=(1, 1),
                    strides=(1, 1), use_bias=True, name='conv3')(x)   
        y = ipt + x * self.beta
        x = nn.LayerNorm()(y)
        x = nn.Dense(features=self.c, name="conv4")(x)
        x = SimpleGate()(x)
        x = nn.Dense(features=self.ffn_channel, name="conv5")(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    img_channel : int = 3
    width : int = 16 #channels
    middle_blk_num : int = 1
    enc_blk_nums : Union[Tuple, List] = []
    dec_blk_nums : Union[Tuple, List] = []

    def setup(self):
        self.padder_size = 2 ** len(enc_blk_nums) 
    @nn.compact
    def __call__(self, inp) :
        B, H, W, C = inp.shape
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size 
        inp = jnp.pad(inp, ((0, 0), (0, mod_pad_h), (0, mod_pad_w), (0, 0)), mode='constant')
        x = nn.Conv(self.width, kernel_size=(3, 3), use_bias=True, name='intro')(inp)
        encs = []
        chan = self.width
        for blk_nums in self.enc_blk_nums :
            x = nn.Sequential([NAFBlock(chan) for _ in range(blk_nums)])(x)
            x = nn.Conv(chan * 2, kernel_size=(2, 2), strides=(2, 2))
            chan = chan * 2
            encs.append(x)    
        x = nn.Sequential([NAFBlock(chan) for _ in range(self.middle_blk_num)])(x)
        for blk_nums, enc_skip in zip(self.dec_blk_nums, encs):
            x = nn.Sequential([nn.Conv(chan * 2, kernel_size=(1, 1), use_bias=False),
                               PixelShuffle(2)])(x)
            x = x + enc_skip
            chan = chan // 2
            x = nn.Sequential([NAFBlock(chan) for _ in range(blk_nums)])(x)
        x = inp + nn.Conv(self.img_channel, kernel_size=(3, 3), use_bias=True, name='ending')(x)
        return jnp.take(x, (B, H, W, C))
