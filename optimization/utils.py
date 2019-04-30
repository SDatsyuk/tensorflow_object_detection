import sys
sys.path.append('..')
from object_detection.protos import input_reader_pb2
from object_detection.protos import model_pb2
from object_detection.protos import pipeline_pb2
from object_detection.protos import train_pb2
from object_detection.utils import config_util
from object_detection.builders import input_reader_builder
from object_detection.builders import model_builder
from object_detection import trainer

import tensorflow as tf
from google.protobuf import text_format
import functools

import os
import json
import random
from collections import namedtuple

Option =  namedtuple('Option', ['name', 'params', 'values', 'format'])

AUGMENTATION_OPTIONS = [
    Option("random_adjust_brightness", ['max_delta'], [[0.1, 0.4]], [float]),
    Option("random_adjust_contrast", ["min_delta", 'max_delta'], [[0.6, 0.8],
    [1.1, 1.4]], [float, float]), Option("random_black_patches", ['max_black_patches', 'probability', "size_to_image_ratio"], [[10, 60], [0.3, 0.7], [0.05, 0.2]], [int, float, float])] # 'normalize_image',

class Model():
    def __init__(self, config, params):
        self.config = config
        self.params = params

    def set_total_loss(self, total_loss):
        self.total_loss = total_loss
 
    def set_eval_metrics(self, metrics):
        self.eval_metrics = metrics


def get_configs_from_pipeline_file(path):
  """Reads training configuration from a pipeline_pb2.TrainEvalPipelineConfig.

  Reads training config from file specified by pipeline_config_path flag.

  Returns:
    model_config: model_pb2.DetectionModel
    train_config: train_pb2.TrainConfig
    input_config: input_reader_pb2.InputReader
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)

  model_config = pipeline_config.model
  train_config = pipeline_config.train_config
  eval_config = pipeline_config.eval_config
  input_config = pipeline_config.train_input_reader

  return model_config, train_config, eval_config, input_config


def get_configs_from_multiple_files(train_config_path, model_config_path, input_config_path):
  """Reads training configuration from multiple config files.

  Reads the training config from the following files:
    model_config: Read from --model_config_path
    train_config: Read from --train_config_path
    input_config: Read from --input_config_path

  Returns:
    model_config: model_pb2.DetectionModel
    train_config: train_pb2.TrainConfig
    input_config: input_reader_pb2.InputReader
  """
  train_config = train_pb2.TrainConfig()
  with tf.gfile.GFile(train_config_path, 'r') as f:
    text_format.Merge(f.read(), train_config)

  model_config = model_pb2.DetectionModel
  with tf.gfile.GFile(model_config_path, 'r') as f:
    text_format.Merge(f.read(), model_config)

  input_config = input_reader_pb2.InputReader()
  with tf.gfile.GFile(input_config_path, 'r') as f:
    text_format.Merge(f.read(), input_config)

  return model_config, train_config, input_config

def update_config(configs):
  print(configs.keys())
  matched_threshold = round(random.uniform(0.3, 0.7), 1)
  unmatched_threshold = round(random.uniform(0.3, matched_threshold), 1)
  dropout_keep_probability = round(random.uniform(0.6, 0.9), 1)
  min_depth = random.randrange(12, 18, 1)
  ssd_generative_params = {
    "model.ssd.matcher.argmax_matcher.matched_threshold": matched_threshold,
    "model.ssd.matcher.argmax_matcher.unmatched_threshold": unmatched_threshold,
    "model.ssd.box_predictor.convolutional_box_predictor.dropout_keep_probability": dropout_keep_probability,
    "model.ssd.feature_extractor.min_depth": min_depth
  }
  
  configs = config_util.merge_external_params_with_configs(configs, kwargs_dict=ssd_generative_params)
  # print(configs['model'].ssd)
  return configs, ssd_generative_params

def shuffle_params(configs, models):
  if configs['model'].HasField('ssd'):
    print(len(models))
    matched_threshold = models[random.choice(list(models.keys()))].params["model.ssd.matcher.argmax_matcher.matched_threshold"]
    unmatched_threshold = min(models[random.choice(list(models.keys()))].params["model.ssd.matcher.argmax_matcher.unmatched_threshold"], matched_threshold)
    ssd_generative_params = {
      "model.ssd.matcher.argmax_matcher.matched_threshold": matched_threshold,
      "model.ssd.matcher.argmax_matcher.unmatched_threshold": unmatched_threshold,
      "model.ssd.box_predictor.convolutional_box_predictor.dropout_keep_probability": models[random.choice(list(models.keys()))].params["model.ssd.box_predictor.convolutional_box_predictor.dropout_keep_probability"],
      "model.ssd.feature_extractor.min_depth": models[random.choice(list(models.keys()))].params["model.ssd.feature_extractor.min_depth"]
      }
    configs = config_util.merge_external_params_with_configs(configs, kwargs_dict=ssd_generative_params)
    # print(configs['model'].ssd)
    return configs, ssd_generative_params

  elif configs['model'].HasField('faster_rcnn'):
    max_total_detections = models[random.choice(list(models.keys()))].params["model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_total_detections"]
    faster_config_params = {
      'model.faster_rcnn.first_stage_nms_iou_threshold': models[random.choice(list(models.keys()))].params["model.faster_rcnn.first_stage_nms_iou_threshold"],
      "model.faster_rcnn.first_stage_max_proposals": models[random.choice(list(models.keys()))].params["model.faster_rcnn.first_stage_max_proposals"],
      "model.faster_rcnn.first_stage_localization_loss_weight": models[random.choice(list(models.keys()))].params["model.faster_rcnn.first_stage_localization_loss_weight"],
      "model.faster_rcnn.second_stage_localization_loss_weight": models[random.choice(list(models.keys()))].params["model.faster_rcnn.second_stage_localization_loss_weight"],
      "model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.score_threshold": models[random.choice(list(models.keys()))].params["model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.score_threshold"],
      "model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.iou_threshold": models[random.choice(list(models.keys()))].params["model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.iou_threshold"],
      "model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_total_detections": max_total_detections,
      "model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_detections_per_class": min(max_total_detections, models[random.choice(list(models.keys()))].params["model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_detections_per_class"])
    }
    configs = config_util.merge_external_params_with_configs(configs, kwargs_dict=faster_config_params)
    return configs, faster_config_params


def update_augmentation_options(config):
  # print(type(config))
  opt = train_pb2.TrainConfig()
  num_of_opt = random.randint(1, len(AUGMENTATION_OPTIONS))
  options = random.sample(AUGMENTATION_OPTIONS, num_of_opt)
  # print(options)
  text = '\n'.join(["data_augmentation_options { %s { %s }}" % (i.name, '\n'.join(["{}: {}".format(o, i.format[j](round(random.uniform(i.values[j][0], i.values[j][1]), 2))) for j, o in enumerate(i.params)])) for i in options])
  text_format.Merge(text, opt)

  if isinstance(config, pipeline_pb2.TrainEvalPipelineConfig):
    # train_config = config.train_config
    config.train_config.MergeFrom(opt)
  elif isinstance(config, train_pb2.TrainConfig):
    # train_config = config
    config.train_config.MergeFrom(opt)
  elif isinstance(config, dict):
    train_config = config['train_config']
    train_config.MergeFrom(opt)
    config['train_config'] = train_config
  else:
    raise AttributeError("Wrong config format. `TrainEvalPipelineConfig` or `TrainConfig` required")

  # print(text)
  return config

def train_process(model_config,
                  input_config,
                  train_config,
                  train_dir,
                  num_clones=1,
                  clone_on_cpu=False):
    model_fn = functools.partial( 
        model_builder.build,
        model_config=model_config,
        is_training=True)

    create_input_dict_fn = functools.partial(
        input_reader_builder.build, input_config)

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    cluster_data = env.get('cluster', None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    task_info = type('TaskSpec', (object,), task_data)

    # Parameters for a single worker.
    ps_tasks = 0
    worker_replicas = 1
    worker_job_name = 'lonely_worker'
    task = 0
    is_chief = True
    master = ''

    if cluster_data and 'worker' in cluster_data:
      # Number of total worker replicas include "worker"s and the "master".
      worker_replicas = len(cluster_data['worker']) + 1
    if cluster_data and 'ps' in cluster_data:
      ps_tasks = len(cluster_data['ps'])

    if worker_replicas > 1 and ps_tasks < 1:
      raise ValueError('At least 1 ps task is needed for distributed training.')

    if worker_replicas >= 1 and ps_tasks > 0:
      # Set up distributed training.
      server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                               job_name=task_info.type,
                               task_index=task_info.index)
      if task_info.type == 'ps':
        server.join()
        return

      worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
      task = task_info.index
      is_chief = (task_info.type == 'master')
      master = server.target

    # change_process_config(os.getpid())
    total_loss = trainer.train(create_input_dict_fn, model_fn, train_config, master, task,
                  num_clones, worker_replicas, clone_on_cpu, ps_tasks,
                  worker_job_name, is_chief, train_dir)
    return total_loss


if __name__ == "__main__":

    config = """
              batch_size: 24
              optimizer {
                rms_prop_optimizer: {
                  learning_rate: {
                    exponential_decay_learning_rate {
                      initial_learning_rate: 0.004
                      decay_steps: 800720
                      decay_factor: 0.95
                    }
                  }
                  momentum_optimizer_value: 0.9
                  decay: 0.9
                  epsilon: 1.0
                }
              }
              fine_tune_checkpoint: 'snapshots/ssd_inception_v2_coco_2017_11_17/model.ckpt'
              from_detection_checkpoint: true
              # Note: The below line limits the training process to 200K steps, which we
              # empirically found to be sufficient enough to train the pets dataset. This
              # effectively bypasses the learning rate schedule (the learning rate will
              # never decay). Remove the below line to train indefinitely.
              num_steps: 200000
              """
    model_config, train_config, eval_config, input_config = get_configs_from_pipeline_file("ssd_mobilenetV1/ssd_mobilenet_v1_coco.config")
    # print(train_config)

    res = update_augmentation_options(train_config)
    print(res)
