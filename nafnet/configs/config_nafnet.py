# Copyright 2021 Google LLC.
#
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

r"""Fine-tunes a Vision Transformer / Hybrid from AugReg checkpoint.

Example for train NAFNet with width=32 on SIDD:

python -m nafnet.main --workdir=./logs \
    --config=$(pwd)/nafnet/configs/config_nafnet.py:32 \
    --config.dataset=SIDD \
    --config.base_lr=0.01

"""

import ml_collections

from nafnet.configs import common
from nafnet.configs import models


def get_config(model):
  """Returns default parameters for finetuning ViT `model` on `dataset`."""
  config = common.get_config()

  if model not in models.MODEL_CONFIGS:
    raise ValueError(f'Unknown Augreg model "{model}"'
                     f'- not found in {set(models.MODEL_CONFIGS.keys())}')
  config.model = models.MODEL_CONFIGS[model].copy_and_resolve_references()

  # These values are often overridden on the command line.
  config.base_lr = 1e-3
  config.total_steps = 200000
  config.warmup_steps = 0
  config.pp = ml_collections.ConfigDict()
  config.pp.train = 'train'
  config.pp.test = 'val'
  config.pp.resize = 256
  config.pp.crop = 256
  config.dataset = '../dataset_unzip/SIDD_Medium_Srgb/'
  # This value MUST be overridden on the command line.

  return config
