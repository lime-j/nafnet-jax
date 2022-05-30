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

import glob
import os
from torchvision.utils import make_grid
import math
from absl import logging
import flax
import jax
import random
from cv2 import rotate
import numpy as np
import tensorflow as tf
import cv2
import sys
if sys.platform != 'darwin':
  # A workaround to avoid crash because tfds may open to many files.
  import resource
  low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

# Adjust depending on the available RAM.
MAX_IN_MEMORY = 200_000



def get_directory_info(directory, ext='[Pp][Nn][Gg]'):
  """Returns information about directory dataset -- see `get_dataset_info()`."""
  inp_glob = f'{directory}/input_crops/*.{ext}'
  gt_glob = f'{directory}/gt_crops/*.{ext}'
  inp_paths = glob.glob(inp_glob)
  gt_paths = glob.glob(gt_glob)
  assert len(inp_paths) == len(gt_paths)
  return dict(
      num_examples=len(inp_paths),
      inp_glob = inp_glob,
      gt_glob = gt_glob,  
  )

def get_dataset_info(dataset, split):
  """Returns information about a dataset.
  
  Args:
    dataset: Name of tfds dataset or directory -- see `./configs/common.py`
    split: Which split to return data for (e.g. "test", or "train"; tfds also
      supports splits like "test[:90%]").

  Returns:
    A dictionary with the following keys:
    - num_examples: Number of examples in dataset/mode.
    - num_classes: Number of classes in dataset.
    - int2str: Function converting class id to class name.
    - examples_glob: Glob to select all files, or None (for tfds dataset).
  """
  directory = os.path.join(dataset, split)
  return get_directory_info(directory)



def get_datasets(config):
  """Returns `ds_train, ds_test` for specified `config`."""
  train_dir = os.path.join(config.dataset, 'train')
  test_dir = os.path.join(config.dataset, 'val')
  if not os.path.isdir(train_dir):
    raise ValueError('Expected to find directories"{}" and "{}"'.format(
        train_dir,
        test_dir, 
    ))
  logging.info('Reading dataset from directories "%s" and "%s"', train_dir,
              test_dir)
  ds_train = get_data_from_directory(
      config=config, directory=train_dir, mode='train')
  ds_test = get_data_from_directory(
      config=config, directory=test_dir, mode='val')
  return ds_train, ds_test


def get_data_from_directory(*, config, directory, mode):
  """Returns dataset as read from specified `directory`."""

  dataset_info = get_directory_info(directory)
  inp_data = tf.data.Dataset.list_files(dataset_info['inp_glob'])
  
  def get_image(path):
      filepath = str(filepath)
      with open(filepath, 'rb') as f:
          value_buf = f.read()
      return imfrombytes(value_buf, float32=True)
  
  def _pp(data):
    return dict(
        input_image = data,
        gt_image = tf.strings.regex_replace(data, "input_crops", "gt_crops")
    )

  image_decoder = lambda path: tf.cast(tf.image.decode_png(tf.io.read_file(path), 3, dtype=tf.uint8), dtype=tf.float32)/ 255.

  return get_data(
      data=inp_data,
      mode=mode,
      image_decoder=image_decoder,
      repeats=None if mode == 'train' else 1,
      batch_size=config.batch_eval if mode == 'val' else config.batch,
      image_size=config.pp['crop'],
      shuffle_buffer=min(dataset_info['num_examples'], config.shuffle_buffer),
      preprocess=_pp)


def get_data(*,
             data,
             mode,
             image_decoder,
             repeats,
             batch_size,
             image_size,
             shuffle_buffer,
             scale = 1,
             preprocess=None):
  """Returns dataset for training/eval.

  Args:
    data: tf.data.Dataset to read data from.
    mode: Must be "train" or "test".
    num_classes: Number of classes (used for one-hot encoding).
    image_decoder: Applied to `features['image']` after shuffling. Decoding the
      image after shuffling allows for a larger shuffle buffer.
    repeats: How many times the dataset should be repeated. For indefinite
      repeats specify None.
    batch_size: Global batch size. Note that the returned dataset will have
      dimensions [local_devices, batch_size / local_devices, ...].
    image_size: Image size after cropping (for training) / resizing (for
      evaluation).
    shuffle_buffer: Number of elements to preload the shuffle buffer with.
    preprocess: Optional preprocess function. This function will be applied to
      the dataset just after repeat/shuffling, and before the data augmentation
      preprocess step is applied.
  """

  def _pp(data):

    ipt_im, gt_im = image_decoder(data['input_image']), image_decoder(data["gt_image"])
    if mode == 'train':
      channels = ipt_im.shape[-1]
      cropped_ims = tf.image.random_crop(tf.concat([gt_im, ipt_im], axis=-1), size=(image_size, image_size, 6))
      gt_im, ipt_im = cropped_ims[:,:,:3], cropped_ims[:,:,3:]
    return {'gt_image': gt_im, 'input_image': ipt_im}

  data = data.repeat(repeats)
  if mode == 'train':
    data = data.shuffle(shuffle_buffer)
  if preprocess is not None:
    data = data.map(preprocess, tf.data.experimental.AUTOTUNE)
  data = data.map(_pp, tf.data.experimental.AUTOTUNE)
  data = data.batch(batch_size, drop_remainder=True)

  # Shard data such that it can be distributed accross devices
  num_devices = jax.local_device_count()

  def _shard(data):
    data['gt_image'] = tf.reshape(data['gt_image'],
                               [num_devices, -1, image_size, image_size, 3])
    data['input_image'] = tf.reshape(data['input_image'],
                               [num_devices, -1, image_size, image_size, 3])
    return data
    
  if num_devices is not None:
    data = data.map(_shard, tf.data.experimental.AUTOTUNE)
  return data


def prefetch(dataset, n_prefetch):
  """Prefetches data to device and converts to numpy array."""
  ds_iter = iter(dataset)
  ds_iter = map(lambda x: jax.tree_map(lambda t: np.asarray(memoryview(t)), x),
                ds_iter)
  if n_prefetch:
   ds_iter = flax.jax_utils.prefetch_to_device(ds_iter, n_prefetch)
  return ds_iter
