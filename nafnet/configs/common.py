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

from typing import Any, Dict, Iterable, Tuple, Union

import ml_collections


def get_config():
  """Returns config values other than model parameters."""

  config = ml_collections.ConfigDict()

  # Where to search for pretrained ViT models.
  # Can be downloaded from gs://vit_models/imagenet21k
  config.pretrained_dir = '.'

  config.dataset = './dataset_unzip/SIDD_Medium_Srgb/'
  # Path to manually downloaded dataset
  config.tfds_manual_dir = './dataset_unzip/SIDD_Medium_Srgb/'
  # Path to tensorflow_datasets directory
  config.tfds_data_dir = None
  # Number of steps; determined by hyper module if not specified.
  config.total_steps = None

  # Resizes global gradients.
  config.grad_norm_clip = 1.0
  # Datatype to use for momentum state ("bfloat16" or "float32").
  config.optim_dtype = 'float32'
  # Accumulate gradients over multiple steps to save on memory.
  config.accum_steps = 1

  # Batch size for training.
  config.batch = 32
  # Batch size for evaluation.
  config.batch_eval = 32
  # Shuffle buffer size.
  config.shuffle_buffer = 50_000
  # Run prediction on validation set every so many steps
  config.eval_every = 1_000
  # Log progress every so many steps.
  config.progress_every = 1_0000
  # How often to write checkpoints. Specifying 0 disables checkpointing.
  config.checkpoint_every = 1_000

  # Number of batches to prefetch to device.
  config.prefetch = 2

  # Base learning-rate for fine-tuning.
  config.base_lr = 0.03
  # How to decay the learning rate ("cosine" or "linear").
  config.decay_type = 'cosine'
  # How to decay the learning rate.
  config.warmup_steps = 500

  # Alternatives : inference_time.
  config.trainer = 'train'

  # Will be set from ./models.py
  config.model = None
  # Only used in ./augreg.py configs
  config.model_or_filename = None
  # Must be set via `with_dataset()`
  config.dataset = None
  config.pp = None

  return config.lock()


# We leave out a subset of training for validation purposes (if needed).
DATASET_PRESETS = {
    'SIDD': ml_collections.ConfigDict(
        {'total_steps': 200_000,
         'pp': ml_collections.ConfigDict(
             {'train': 'train',
              'test': 'test',
              'crop': 256})
         }),
}


def with_dataset(config: ml_collections.ConfigDict,
                 dataset: str) -> ml_collections.ConfigDict:
  config = ml_collections.ConfigDict(config.to_dict())
  config.dataset = dataset
  config.update(DATASET_PRESETS[dataset])
  return config


def flatten(
    config: Union[ml_collections.ConfigDict, Dict[str, Any]],
    prefix: Tuple[str, ...] = ('config',)
) -> Iterable[Tuple[str, Any]]:
  """Returns a flat representation of `config`, e.g. for use in sweeps."""
  for k, v in config.items():
    if isinstance(v, (dict, ml_collections.ConfigDict)):
      yield from flatten(v, prefix + (k,))
    else:
      yield ('.'.join(prefix + (k,)), v)
