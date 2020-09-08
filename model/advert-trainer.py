import os
import absl
import datetime
import tensorflow as tf
import tensorflow_transform as tft

from tfx.components.trainer.executor import TrainerFnArgs

from model import constants

_DENSE_FLOAT_FEATURE_KEYS = constants.DENSE_FLOAT_FEATURE_KEYS
_VOCAB_FEATURE_KEYS = constants.VOCAB_FEATURE_KEYS
_VOCAB_SIZE = constants.VOCAB_SIZE
_OOV_SIZE = constants.OOV_SIZE
_LABEL_KEY = constants.LABEL_KEY
_transformed_name = constants.transformed_name


def _transformed_names(keys):
  return [_transformed_name(key) for key in keys]


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(
      filenames,
      compression_type='GZIP')


def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example and applies TFT."""

  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    transformed_features = model.tft_layer(parsed_features)
    return model(transformed_features)

  return serve_tf_examples_fn


def _input_fn(file_pattern, tf_transform_output,
              batch_size=100):
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      label_key=_transformed_name(_LABEL_KEY))

  return dataset


def _build_keras_model(hidden_units):
  """Creates a DNN Keras model for classifying taxi data.
  """
  real_valued_columns = [
      tf.feature_column.numeric_column(key, shape=())
      for key in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
  ]
  categorical_columns = [
      tf.feature_column.categorical_column_with_identity(
          key, num_buckets=_VOCAB_SIZE + _OOV_SIZE, default_value=0)
      for key in _transformed_names(_VOCAB_FEATURE_KEYS)
  ]
  indicator_column = [
      tf.feature_column.indicator_column(categorical_column)
      for categorical_column in categorical_columns
  ]

  model = _wide_and_deep_classifier(
      wide_columns=indicator_column,
      deep_columns=real_valued_columns,
      dnn_hidden_units=hidden_units or [100, 70, 60, 50])
  return model


def _wide_and_deep_classifier(wide_columns, deep_columns, dnn_hidden_units):
  """returns a simple keras wide and deep model.
  """
  input_layers = {
      colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)
      for colname in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
  }
  
  input_layers.update({
      colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
      for colname in _transformed_names(_VOCAB_FEATURE_KEYS)
  })

  deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)
  for numnodes in dnn_hidden_units:
    deep = tf.keras.layers.Dense(numnodes)(deep)
    
  wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)

  output = tf.keras.layers.Dense(
      1, activation='sigmoid')(
          tf.keras.layers.concatenate([deep, wide]))

  model = tf.keras.Model(input_layers, output)
  model.compile(
      loss='binary_crossentropy',
      optimizer=tf.keras.optimizers.Adam(lr=0.01),
      metrics=[tf.keras.metrics.BinaryAccuracy()])
  model.summary(print_fn=absl.logging.info)
  return model



# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.
  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  # Number of nodes in the first layer of the DNN
  first_dnn_layer_size = 150
  num_dnn_layers = 4
  dnn_decay_factor = 0.7

  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 40)
  eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 40)

  model = _build_keras_model(
      # Construct layers sizes with exponetial decay
      hidden_units=[
          max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
          for i in range(num_dnn_layers)
      ])

  log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir, update_freq='batch')
  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }

  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)