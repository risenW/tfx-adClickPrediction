import tensorflow as tf
import tensorflow_transform as tft

from model import constants

_DENSE_FLOAT_FEATURE_KEYS = constants.DENSE_FLOAT_FEATURE_KEYS
_LABEL_KEY = constants.LABEL_KEY
_VOCAB_FEATURE_KEYS = constants.VOCAB_FEATURE_KEYS
_VOCAB_SIZE = constants.VOCAB_SIZE
_OOV_SIZE = constants.OOV_SIZE
_transformed_name = constants.transformed_name


def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.
  Args:
    inputs: map from feature keys to raw not-yet-transformed features.
  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}
  for key in _DENSE_FLOAT_FEATURE_KEYS:
    # Preserve this feature as a dense float, setting nan's to the mean.
    outputs[_transformed_name(key)] = tft.scale_to_z_score(
        inputs[key])
    
  for key in _VOCAB_FEATURE_KEYS:
    # Build a vocabulary for this feature.
    outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
        inputs[key],
        top_k=_VOCAB_SIZE,
        num_oov_buckets=_OOV_SIZE)

  outputs[_transformed_name(_LABEL_KEY)] = inputs[_LABEL_KEY]

  return outputs
