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

def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.
    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.
    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED
    }
    if img_np is None:
        raise Exception('None .. !!!')
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.
    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.
    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tf.convert_to_tensor(img.transpose(2, 0, 1), dtype=tf.float32 if float32 else tf.uint8)
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def padding(img_lq, img_gt, gt_size):
    h, w, _ = img_lq.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    
    if h_pad == 0 and w_pad == 0:
        return img_lq, img_gt

    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    # print('img_lq', img_lq.shape, img_gt.shape)
    return img_lq, img_gt

def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale):
    """Paired random crop.
    It crops lists of lq and gt images with corresponding locations.
    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.
    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). ')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def get_directory_info(inp_dir, gt_dir, ext='jpg'):
  """Returns information about directory dataset -- see `get_dataset_info()`."""
  inp_glob = f'{directory}/*/*.{ext}'
  gt_glob = f'{directory}/*/*.{ext}'
  inp_paths = glob.glob(examples_glob)
  gt_paths = glob.glob(examples_glob)
  assert len(inp_paths) == len(gt_paths)
  return dict(
      num_examples=len(paths),
      inp_glob = inp_glob,
      gt_glob = gt_glob,  
  )



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
  gt_data = tf.data.Dataset.list_files(dataset_info['gt_glob'])
  
  def get_image(path):
      filepath = str(filepath)
      with open(filepath, 'rb') as f:
          value_buf = f.read()
      return imfrombytes(value_buf, float32=True)
  
  def _pp(data):
    return dict(
        image = data,
    )

  image_decoder = lambda path: get_image(path)

  return get_data(
      inp_data = inp_data,
      gt_data = gt_data,
      mode=mode,
      num_classes=dataset_info['num_classes'],
      image_decoder=image_decoder,
      repeats=None if mode == 'train' else 1,
      batch_size=config.batch_eval if mode == 'val' else config.batch,
      image_size=config.pp['crop'],
      scale = config.scale,
      shuffle_buffer=min(dataset_info['num_examples'], config.shuffle_buffer),
      preprocess=_pp)


def get_data(*,
             inp_data,
             gt_data,
             mode,
             num_classes,
             image_decoder,
             repeats,
             batch_size,
             image_size,
             shuffle_buffer,
             scale,
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

  def _pp(inp_data, gt_data):

    ipt_im, gt_im = image_decoder(os.join(ipt_data['image'], 'input_crops')), image_decoder(os.join(gt_data['image'], 'gt_crops'))
    if ipt_im.shape[-1] == 1: im = np.concatenate([im, im, im], axis=-1)
    if gt_im.shape[-1] == 1: gt_im = np.concatenate([gt_im, gt_im, gt_im], axis=-1)

    if mode == 'train':
      channels = ipt_im.shape[-1]
      gt_im, ipt_im = padding(gt_im, ipt_im, image_size)
      gt_im, ipt_im = paired_random_crop(gt_im, ipt_im, image_size, scale)
      #gt_im, ipt_im = augment([img_gt, img_lq], self.opt['use_flip'],
      #                    self.opt['use_rot'])
      
    gt_im, ipt_im = img2tensor([gt_im, ipt_im], bgr2rgb=True, float32=True)
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

  if num_devices is not None and mode == 'train':
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
