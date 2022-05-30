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

import ml_collections

# The key of this dictionary refers to basename in the directory:
# https://console.cloud.google.com/storage/vit_models/
# Note that some names (e.g. "testing", but also some models only available in
# the AugReg paper) are not actually present in that directory.
MODEL_CONFIGS = {}


def _register(get_config):
  """Adds reference to model config into MODEL_CONFIGS and AUGREG_CONFIGS."""
  config = get_config().lock()
  name = config.get('name')
  MODEL_CONFIGS[name] = config
  return get_config


@_register
def get_w32_config():
  """Returns the width=32 NAFNet configuration."""
  config = ml_collections.ConfigDict()
  config.name = 'w32'
  config.img_channel = 3
  config.width = 32
  config.middle_blk_num = 12
  config.enc_blk_nums = [2, 2, 4, 8]
  config.dec_blk_nums = [2, 2, 2, 2]

  return config

@_register
def get_w64_config():
  """Returns the width=64 NAFNet configuration."""
  config = ml_collections.ConfigDict()
  config.name = 'w64'
  config.img_channel = 3
  config.width = 64
  config.middle_blk_num = 12
  config.enc_blk_nums = [2, 2, 4, 8]
  config.dec_blk_nums = [2, 2, 2, 2]

  return config
