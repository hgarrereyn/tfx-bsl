
import argparse
import os

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
import tfx_bsl
from tfx_bsl.public.proto import model_spec_pb2
from  tfx_bsl.public.beam.run_inference import RunInference
import tensorflow as tf

from absl import logging
logging.set_verbosity(logging.INFO)


def get_saved_model_spec(model_path):
  '''Returns an InferenceSpecType object for a saved model path.'''
  return model_spec_pb2.InferenceSpecType(
    saved_model_spec=model_spec_pb2.SavedModelSpec(
        model_path=model_path))

def make_example(x):
  '''Build a TFExample object to feed to the model.'''
  feature = {}
  feature['input'] = tf.train.Feature(
      float_list=tf.train.FloatList(value=[x]))
  ex = tf.train.Example(features=tf.train.Features(feature=feature))
  return ex

def run():
    '''Run demo pipeline.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='''
    Location of a model
    ''')
    parser.add_argument('tfx_bsl', help='''
    Path to tfx bsl .whl file
    ''')
    known_args, pipeline_args = parser.parse_known_args()

    options = PipelineOptions(pipeline_args)

    setup_options = options.view_as(SetupOptions)
    setup_options.extra_packages = [known_args.tfx_bsl]
    setup_options.save_main_session = True

    p = beam.Pipeline(options=options)

    inference_spec = get_saved_model_spec(known_args.model_path)
    examples = p | 'Create examples' >> beam.Create([make_example(5)])

    _ = examples | 'Run inference' >> RunInference(inference_spec) 

    result = p.run()
    result.wait_until_finish()

if __name__=='__main__':
    run()
