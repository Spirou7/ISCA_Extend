# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Keras image preprocessing layers."""

import tensorflow.compat.v2 as tf
import numpy as np
import numbers

# Updated imports for TensorFlow 2.16+ and Keras 3 compatibility
from tensorflow.keras import backend
from tensorflow.keras.layers import Layer
from tensorflow.keras import utils as keras_utils

# Internal TensorFlow APIs. It's generally recommended to use public APIs,
# but for this file's functionality, some are retained.
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops

ResizeMethod = tf.image.ResizeMethod

_RESIZE_METHODS = {
    'bilinear': ResizeMethod.BILINEAR,
    'nearest': ResizeMethod.NEAREST_NEIGHBOR,
    'bicubic': ResizeMethod.BICUBIC,
    'area': ResizeMethod.AREA,
    'lanczos3': ResizeMethod.LANCZOS3,
    'lanczos5': ResizeMethod.LANCZOS5,
    'gaussian': ResizeMethod.GAUSSIAN,
    'mitchellcubic': ResizeMethod.MITCHELLCUBIC
}

H_AXIS = -3
W_AXIS = -2

def get_rotation_matrix(angles, image_height, image_width):
    """Returns a 3x3 transformation matrix for rotating images."""
    image_height = tf.cast(image_height, dtype=tf.float32)
    image_width = tf.cast(image_width, dtype=tf.float32)
    cos_angles = tf.cos(angles)
    sin_angles = tf.sin(angles)
    x_offset = ((image_width - 1) - (cos_angles * (image_width - 1) - sin_angles *
                                     (image_height - 1))) / 2.0
    y_offset = ((image_height - 1) - (sin_angles * (image_width - 1) + cos_angles *
                                      (image_height - 1))) / 2.0
    return tf.convert_to_tensor([
        cos_angles, -sin_angles, x_offset, sin_angles, cos_angles, y_offset,
        0.0, 0.0, 1.0
    ], dtype=tf.float32)

def transform(images,
              transforms,
              fill_mode='reflect',
              fill_value=0.0,
              interpolation='bilinear',
              output_shape=None):
    """Applies the given transform to a batch of images."""
    if output_shape is None:
        output_shape = tf.shape(images)[1:3]
    if not tf.is_tensor(output_shape):
        output_shape = tf.convert_to_tensor(output_shape, tf.int32)
    if len(output_shape.shape) == 0:
        output_shape = tf.stack([output_shape, output_shape])

    return tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=transforms,
        output_shape=output_shape,
        fill_value=fill_value,
        interpolation=interpolation.upper(),
        fill_mode=fill_mode.upper())

def check_fill_mode_and_interpolation(fill_mode, interpolation):
  if fill_mode not in {'reflect', 'wrap', 'constant', 'nearest'}:
    raise NotImplementedError(
        f'Unknown `fill_mode` {fill_mode}. Only `reflect`, `wrap`, '
        '`constant` and `nearest` are supported.')
  if interpolation not in {'nearest', 'bilinear'}:
    raise NotImplementedError(
        f'Unknown `interpolation` {interpolation}. Only `nearest` and '
        '`bilinear` are supported.')

def make_generator(seed=None):
  """Creates a random generator."""
  if seed is not None:
    return tf.random.Generator.from_seed(seed)
  else:
    return tf.random.Generator.from_non_deterministic_state()

class MyRandomCrop(Layer):
  """Randomly crop the images to target height and width."""
  def __init__(self, height, width, seed=None, **kwargs):
    super(MyRandomCrop, self).__init__(**kwargs)
    self.height = height
    self.width = width
    self.seed = seed
    self._rng = make_generator(self.seed)

  def call(self, inputs, training=True):
    if training is None:
      training = backend.learning_phase()

    inputs = tf.convert_to_tensor(inputs)
    unbatched = inputs.shape.rank == 3

    def random_cropped_inputs():
      """Cropped inputs with stateless random ops."""
      shape = tf.shape(inputs)
      if unbatched:
        crop_size = tf.stack([self.height, self.width, shape[-1]])
      else:
        crop_size = tf.stack([shape[0], self.height, self.width, shape[-1]])
      
      check = tf.Assert(
          tf.reduce_all(shape >= crop_size),
          [self.height, self.width])
      
      with tf.control_dependencies([check]):
        limit = shape - crop_size + 1
        offset = tf.random.stateless_uniform(
            tf.shape(shape),
            dtype=crop_size.dtype,
            maxval=crop_size.dtype.max,
            seed=self._rng.make_seeds()[:, 0]) % limit
        return tf.slice(inputs, offset, crop_size)

    def resize_and_center_cropped_inputs():
      """Deterministically resize to shorter side and center crop."""
      input_shape = tf.shape(inputs)
      input_height_t = input_shape[H_AXIS]
      input_width_t = input_shape[W_AXIS]
      ratio_cond = (input_height_t / input_width_t > (self.height / self.width))
      
      resized_height = keras_utils.smart_cond(
          ratio_cond,
          lambda: tf.cast(self.width * input_height_t / input_width_t,
                          input_height_t.dtype), lambda: self.height)
      resized_width = keras_utils.smart_cond(
          ratio_cond, lambda: self.width,
          lambda: tf.cast(self.height * input_width_t / input_height_t,
                          input_width_t.dtype))
      
      resized_inputs = tf.image.resize(
          images=inputs, size=tf.stack([resized_height, resized_width]))

      img_hd_diff = resized_height - self.height
      img_wd_diff = resized_width - self.width
      bbox_h_start = tf.cast(img_hd_diff / 2, tf.int32)
      bbox_w_start = tf.cast(img_wd_diff / 2, tf.int32)
      if unbatched:
        bbox_begin = tf.stack([bbox_h_start, bbox_w_start, 0])
        bbox_size = tf.stack([self.height, self.width, -1])
      else:
        bbox_begin = tf.stack([0, bbox_h_start, bbox_w_start, 0])
        bbox_size = tf.stack([-1, self.height, self.width, -1])
      outputs = tf.slice(resized_inputs, bbox_begin, bbox_size)
      return outputs

    output = keras_utils.smart_cond(training, random_cropped_inputs,
                                          resize_and_center_cropped_inputs)
    input_shape = inputs.shape.as_list()
    if unbatched:
      output_shape = [self.height, self.width, input_shape[-1]]
    else:
      output_shape = [input_shape[0], self.height, self.width, input_shape[-1]]
    output.set_shape(output_shape)
    return output

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    input_shape[H_AXIS] = self.height
    input_shape[W_AXIS] = self.width
    return tf.TensorShape(input_shape)

  def get_config(self):
    config = {
        'height': self.height,
        'width': self.width,
        'seed': self.seed,
    }
    base_config = super(MyRandomCrop, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

HORIZONTAL = 'horizontal'
VERTICAL = 'vertical'
HORIZONTAL_AND_VERTICAL = 'horizontal_and_vertical'

class MyRandomFlip(Layer):
  """Randomly flip each image horizontally and vertically."""
  def __init__(self,
               mode=HORIZONTAL_AND_VERTICAL,
               seed=None,
               **kwargs):
    super(MyRandomFlip, self).__init__(**kwargs)
    self.mode = mode
    if mode == HORIZONTAL:
      self.horizontal = True
      self.vertical = False
    elif mode == VERTICAL:
      self.horizontal = False
      self.vertical = True
    elif mode == HORIZONTAL_AND_VERTICAL:
      self.horizontal = True
      self.vertical = True
    else:
      raise ValueError(f'RandomFlip layer {self.name} received an unknown mode '
                       f'argument {mode}')
    self.seed = seed
    self._rng = make_generator(self.seed)

  def call(self, inputs, training=True):
    if training is None:
      training = backend.learning_phase()

    def random_flipped_inputs():
      flipped_outputs = inputs
      if self.horizontal:
        flipped_outputs = tf.image.stateless_random_flip_left_right(
            flipped_outputs,
            self._rng.make_seeds()[:, 0])
      if self.vertical:
        flipped_outputs = tf.image.stateless_random_flip_up_down(
            flipped_outputs,
            self._rng.make_seeds()[:, 0])
      return flipped_outputs

    output = keras_utils.smart_cond(training, random_flipped_inputs,
                                          lambda: inputs)
    output.set_shape(inputs.shape)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'mode': self.mode,
        'seed': self.seed,
    }
    base_config = super(MyRandomFlip, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class MyRandomRotation(Layer):
  """Randomly rotate each image."""
  def __init__(self,
               factor,
               fill_mode='reflect',
               interpolation='bilinear',
               seed=None,
               fill_value=0.0,
               **kwargs):
    super(MyRandomRotation, self).__init__(**kwargs)
    self.factor = factor
    if isinstance(factor, (tuple, list)):
      self.lower = factor[0]
      self.upper = factor[1]
    else:
      self.lower = -factor
      self.upper = factor
    if self.upper < self.lower:
      raise ValueError(f'Factor cannot have negative values, got {factor}')
    check_fill_mode_and_interpolation(fill_mode, interpolation)
    self.fill_mode = fill_mode
    self.fill_value = fill_value
    self.interpolation = interpolation
    self.seed = seed
    self._rng = make_generator(self.seed)

  def call(self, inputs, training=True):
    if training is None:
      training = backend.learning_phase()

    inputs = tf.convert_to_tensor(inputs)
    original_shape = inputs.shape
    unbatched = inputs.shape.rank == 3
    if unbatched:
      inputs = tf.expand_dims(inputs, 0)

    def random_rotated_inputs():
      """Rotated inputs with random ops."""
      inputs_shape = tf.shape(inputs)
      batch_size = inputs_shape[0]
      img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
      img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)
      min_angle = self.lower * 2. * np.pi
      max_angle = self.upper * 2. * np.pi
      angles = self._rng.uniform(
          shape=[batch_size], minval=min_angle, maxval=max_angle)
      return transform(
          inputs,
          get_rotation_matrix(angles, img_hd, img_wd),
          fill_mode=self.fill_mode,
          fill_value=self.fill_value,
          interpolation=self.interpolation)

    output = keras_utils.smart_cond(training, random_rotated_inputs,
                                          lambda: inputs)
    if unbatched:
      output = tf.squeeze(output, 0)
    output.set_shape(original_shape)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'factor': self.factor,
        'fill_mode': self.fill_mode,
        'fill_value': self.fill_value,
        'interpolation': self.interpolation,
        'seed': self.seed,
    }
    base_config = super(MyRandomRotation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

def get_interpolation(interpolation):
  interpolation = interpolation.lower()
  if interpolation not in _RESIZE_METHODS:
    raise NotImplementedError(
        f'Value not recognized for `interpolation`: {interpolation}. Supported values '
        f'are: {_RESIZE_METHODS.keys()}')
  return _RESIZE_METHODS[interpolation]

def _get_noise_shape(x, noise_shape):
  if noise_shape is None:
    return array_ops.shape(x)
  try:
    noise_shape_ = tensor_shape.as_shape(noise_shape)
  except (TypeError, ValueError):
    return noise_shape

  if x.shape.dims is not None and len(x.shape.dims) == len(noise_shape_.dims):
    new_dims = []
    for i, dim in enumerate(x.shape.dims):
      if noise_shape_.dims[i].value is None and dim.value is not None:
        new_dims.append(dim.value)
      else:
        new_dims.append(noise_shape_.dims[i].value)
    return tensor_shape.TensorShape(new_dims)
  return noise_shape

def my_dropout(x, rate, noise_shape=None, seed=None, name=None):
   """Computes dropout: randomly sets elements to zero to prevent overfitting."""
   if x.get_shape().as_list()[0] is None:
       return x
   if seed is None:
      seed = [np.random.randint(10e6), np.random.randint(10e6)]
   else:
      seed = [seed, seed+1]
   with ops.name_scope(name, "dropout", [x]) as name:
    is_rate_number = isinstance(rate, numbers.Real)
    if is_rate_number and (rate < 0 or rate >= 1):
      raise ValueError(f"rate must be a scalar tensor or a float in the range [0, 1), got {rate}")
    
    x = ops.convert_to_tensor(x, name="x")
    x_dtype = x.dtype
    if not x_dtype.is_floating:
      raise ValueError(f"x has to be a floating point tensor since it's going to be scaled. Got a {x_dtype} tensor instead.")
    
    if is_rate_number and rate == 0:
       random_seed.get_seed(seed)
       return x

    is_executing_eagerly = context.executing_eagerly()
    if not tensor_util.is_tf_type(rate):
      if is_rate_number:
        keep_prob = 1 - rate
        scale = 1 / keep_prob
        scale = ops.convert_to_tensor(scale, dtype=x_dtype)
        ret = gen_math_ops.mul(x, scale)
      else:
        raise ValueError(f"rate is neither scalar nor scalar tensor {rate!r}")
    else:
      rate.get_shape().assert_has_rank(0)
      rate_dtype = rate.dtype
      if not rate_dtype.is_compatible_with(x_dtype):
        raise ValueError(
           f"Tensor dtype {x_dtype.name} is incompatible with Tensor dtype {rate_dtype.name}: {rate!r}")
      rate = gen_math_ops.cast(rate, x_dtype, name="rate")
      one_tensor = constant_op.constant(1, dtype=x_dtype)
      ret = gen_math_ops.real_div(x, gen_math_ops.sub(one_tensor, rate))

    noise_shape = _get_noise_shape(x, noise_shape)
    
    # Use tf.random.stateless_uniform for reproducibility
    random_tensor = tf.random.stateless_uniform(
        noise_shape, seed=seed, dtype=x_dtype)
    
    keep_mask = random_tensor >= rate
    ret = gen_math_ops.mul(ret, gen_math_ops.cast(keep_mask, x_dtype))
    if not is_executing_eagerly:
      ret.set_shape(x.get_shape())
    return ret