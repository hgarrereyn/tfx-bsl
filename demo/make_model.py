import argparse
import os

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
import tfx_bsl
from tfx_bsl.public.proto import model_spec_pb2
from  tfx_bsl.public.beam.run_inference import RunInference
import tensorflow as tf


def build_keras_model():
  '''Build a dummy keras model with one input and output.'''
  inp = tf.keras.layers.Input(shape=(1,), name='input')
  out = tf.keras.layers.Dense(1, name='output')(inp)
  return tf.keras.models.Model(inp, out)

class WrapKerasModel(tf.keras.Model):
  '''Wrapper class to apply a signature to a keras model.'''
  def __init__(self, model):
    super().__init__()
    self.model = model

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='inputs')
  ])
  def call(self, serialized_example):
    features = {
      'input': tf.compat.v1.io.FixedLenFeature(
        [1],
        dtype=tf.float32,
        default_value=0
      )
    }
    input_tensor_dict = tf.io.parse_example(serialized_example, features)
    return self.model(input_tensor_dict)

def save_keras_model(model, path):
  '''Export a keras model in the SavedModel format.'''
  wrapped_model = WrapKerasModel(model)
  tf.compat.v1.keras.experimental.export_saved_model(
    wrapped_model, path, serving_only=True
  )

def run():
    '''Run demo pipeline.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='''
    A directory in a gcp bucket to store the model
    ''')
    known_args, pipeline_args = parser.parse_known_args()

    # create and save a demo model
    m = build_keras_model()
    save_keras_model(m, known_args.model_path)
    print(f'Saved model at {known_args.model_path}')

if __name__=='__main__':
    run()
