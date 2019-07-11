# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Prepares the data used for DeepLab training/evaluation."""
import tensorflow as tf
from deeplab.core import feature_extractor
from deeplab.core import preprocess_utils

# The probability of flipping the images and labels
# left-right during training
_PROB_OF_FLIP = 0.5

FLAGS = tf.app.flags.FLAGS


def preprocess_image_and_label(image,
                               label,
                               crop_height,
                               crop_width,
                               min_resize_value=None,
                               max_resize_value=None,
                               resize_factor=None,
                               min_scale_factor=1.,
                               max_scale_factor=1.,
                               scale_factor_step_size=0,
                               background_label=0,
                               is_training=True,
                               model_variant=None,
                               labels_multichannel=None):
  """Preprocesses the image and label.

  Args:
    image: Input image.
    label: Ground truth annotation label.
    crop_height: The height value used to crop the image and label.
    crop_width: The width value used to crop the image and label.
    min_resize_value: Desired size of the smaller image side.
    max_resize_value: Maximum allowed size of the larger image side.
    resize_factor: Resized dimensions are multiple of factor plus one.
    min_scale_factor: Minimum scale factor value.
    max_scale_factor: Maximum scale factor value.
    scale_factor_step_size: The step size from min scale factor to max scale
      factor. The input is randomly scaled based on the value of
      (min_scale_factor, max_scale_factor, scale_factor_step_size).
    ignore_label: The label value which will be ignored for training and
      evaluation.
    is_training: If the preprocessing is used for training or not.
    model_variant: Model variant (string) for choosing how to mean-subtract the
      images. See feature_extractor.network_map for supported model variants.

  Returns:
    original_image: Original image (could be resized).
    processed_image: Preprocessed image.
    label: Preprocessed ground truth segmentation label.

  Raises:
    ValueError: Ground truth label not provided during training.
  """
  if is_training and (label is None or labels_multichannel is None):
    raise ValueError('During training, label must be provided.')
  if model_variant is None:
    tf.logging.warning('Default mean-subtraction is performed. Please specify '
                       'a model_variant. See feature_extractor.network_map for '
                       'supported model variants.')

  # Keep reference to original image.
  original_image = image
  processed_image = tf.cast(image, tf.float32)

  if label is not None:
    label = tf.cast(label, tf.int32)

  if labels_multichannel is not None:
    # convert from range [0,255], to range [0,1]
    labels_multichannel = tf.cast(labels_multichannel, tf.float32)
    labels_multichannel /= 255.0

  if is_training:
    # Randomly left-right flip the image and label.

    #skip flip because corner labels
    processed_image, label, labels_multichannel, is_flipped = preprocess_utils.flip_dim(
        [processed_image, label, labels_multichannel], _PROB_OF_FLIP, dim=1, left_labels=FLAGS.left_labels, right_labels=FLAGS.right_labels)

    #randomly up-down flip the image and label
    #useful when training the same model both for portrait and landscape mode
    #when part of the training set is 90 degrees rotated
    #processed_image, label, _ = preprocess_utils.flip_dim(
    #    [processed_image, label], _PROB_OF_FLIP, dim=0)

    #random_gamma (tf.image.adjust_gamma doesn't work right - it gets the uppe rlimit (255) wrong)
    random_gamma = 1.0 / tf.random_uniform([1],0.4, 1.6)[0]
    processed_image = (processed_image / 255.0) ** random_gamma * 255.

    #random_hue
    processed_image = tf.image.random_hue(processed_image, 0.5)

    #random_light_color
    processed_image *= [tf.random_uniform([1],0.8, 1.2)[0], tf.random_uniform([1],0.8, 1.2)[0], tf.random_uniform([1],0.8, 1.2)[0]]

    #random contrast & brightness:
    #tensorflow random_brightness is doing it seperately for each channel
    #i want the same transformation for all channels
    #first do the contrast, then do brightness trying to fit in 0-255 range and not clip
    grayscale = tf.reduce_sum(processed_image * [0.299, .587, .114], axis=2)
    mean = tf.reduce_mean(grayscale)
    min = tf.reduce_min(grayscale)
    max = tf.reduce_max(grayscale)
    random_contrast = tf.random_uniform([1],0.6, 1.1) #bias to lower contrast (more similar to real data)
    brightness_min = -(mean - ((mean - min) * random_contrast))
    brightness_max = 255.0 - ((max - mean) * random_contrast + mean)
    random_brightness = tf.random_uniform([1],brightness_min, brightness_max)
    processed_image -= [mean, mean, mean]
    processed_image *= [random_contrast[0], random_contrast[0], random_contrast[0]]
    processed_image += [mean + random_brightness[0], mean + random_brightness[0], mean + random_brightness[0]]

    #random blur every 5th sample
    processed_image = tf.cond(
      tf.greater(tf.random_uniform([1],0, 1)[0], 0.1),
      lambda: processed_image,
      lambda: preprocess_utils.gaussian_blur(processed_image)
    )

    #there won't be overflows in real life
    processed_image = tf.clip_by_value(processed_image, 0.0, 255.0)


  # Resize image and label to the desired range.
  # should resize also labels_multichannel, but i am not really using resize during training
  if min_resize_value is not None or max_resize_value is not None:
    [processed_image, label] = (
        preprocess_utils.resize_to_range(
            image=processed_image,
            label=label,
            min_size=min_resize_value,
            max_size=max_resize_value,
            factor=resize_factor,
            align_corners=True))
    # The `original_image` becomes the resized image.
    original_image = tf.identity(processed_image)

  # Data augmentation by randomly scaling the inputs.
  if is_training:
    scale = preprocess_utils.get_random_scale(
        min_scale_factor, max_scale_factor, scale_factor_step_size)
    #should scale also labels_multichannel, but i am not currently using scale during training
    processed_image, label = preprocess_utils.randomly_scale_image_and_label(
        processed_image, label, scale)
    processed_image.set_shape([None, None, 3])

  # Pad image and label to have dimensions >= [crop_height, crop_width]
  image_shape = tf.shape(processed_image)
  image_height = image_shape[0]
  image_width = image_shape[1]

  target_height = image_height + tf.maximum(crop_height - image_height, 0)
  target_width = image_width + tf.maximum(crop_width - image_width, 0)

  # Pad image with mean pixel value.
  mean_pixel = tf.reshape(
      feature_extractor.mean_pixel(model_variant), [1, 1, 3])
  processed_image = preprocess_utils.pad_to_bounding_box(
      processed_image, 0, 0, target_height, target_width, mean_pixel)

  if label is not None:
  #should also pad labels_multichannel, but my generated training data are guaranteed to be at the right size
    label = preprocess_utils.pad_to_bounding_box(
        label, 0, 0, target_height, target_width, background_label)

  # Randomly crop the image and label.
  if is_training and label is not None:
    processed_image, label, labels_multichannel = preprocess_utils.random_crop(
        [processed_image, label, labels_multichannel], crop_height, crop_width)

  processed_image.set_shape([crop_height, crop_width, 3])

  if label is not None:
    label.set_shape([crop_height, crop_width, 1])

  if labels_multichannel is  not None:
    labels_multichannel.set_shape([crop_height, crop_width, FLAGS.num_classes])

  # switch to range [-1;1] normally happens in feature extractor, but it won't work with quantization
  if FLAGS.input_floats:
    processed_image = (2.0 / 255.0) * processed_image - 1.0

  return original_image, processed_image, label, labels_multichannel

