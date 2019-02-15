import tensorflow as tf
import functools
import json
import os
import configparser
import random
# import logging
import sys
import pickle

from google.protobuf import text_format

sys.path.append("..")
from model_evaluator import _run_checkpoint_once, get_evaluators


from object_detection import evaluator
from object_detection.builders import input_reader_builder
from object_detection.builders import model_builder

from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_utils
from object_detection import eval_util
from object_detection.utils import config_util

from utils import get_configs_from_pipeline_file, update_config, update_augmentation_options, Model, shuffle_params, train_process

# logging.basicConfig(level=logging.INFO)
# tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.propagate = False

train_steps = 2000
gen_iter = 2
eval_metrics = 'pascal_voc_detection_metrics'
top = 4

flags = tf.app.flags

flags.DEFINE_string('master', '', 'BNS name of the TensorFlow master to use.')
flags.DEFINE_integer('task', 0, 'task id')
flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy per worker.')
flags.DEFINE_boolean('clone_on_cpu', False,
                     'Force clones to be deployed on CPU.  Note that even if '
                     'set to False (allowing ops to run on gpu), some ops may '
                     'still be run on the CPU if they have no GPU kernel.')
flags.DEFINE_integer('worker_replicas', 1, 'Number of worker+trainer '
                     'replicas.')
flags.DEFINE_integer('ps_tasks', 0,
                     'Number of parameter server tasks. If None, does not use '
                     'a parameter server.')
flags.DEFINE_string('train_dir', '',
                    'Directory to save the checkpoints and training summaries.')

flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')

flags.DEFINE_string('train_config_path', '',
                    'Path to a train_pb2.TrainConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')

FLAGS = flags.FLAGS


def evaluate(config, train_dir, eval_dir):

  model_config, eval_config, input_config = config
  
  create_input_dict_fn = functools.partial(
        input_reader_builder.build,
        input_config )

  model = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=False )()

  tensor_dict, losses_dict = evaluator._extract_predictions_and_losses(
      model=model,
      create_input_dict_fn=create_input_dict_fn,
      ignore_groundtruth=eval_config.ignore_groundtruth )

  label_map = label_map_util.load_labelmap( input_config.label_map_path )
  max_num_classes = max( [item.id for item in label_map.item] )
  categories = label_map_util.convert_label_map_to_categories(
      label_map, max_num_classes )

  evaluator_list = get_evaluators( eval_config, categories, eval_metrics)

  checkpoint_dir = train_dir
  variables_to_restore = None

  def _process_batch(tensor_dict, sess, batch_index, counters,
                     losses_dict=None):
      """Evaluates tensors in tensor_dict, losses_dict and visualizes examples.

    This function calls sess.run on tensor_dict, evaluating the original_image
    tensor only on the first K examples and visualizing detections overlaid
    on this original_image.

    Args:
      tensor_dict: a dictionary of tensors
      sess: tensorflow session
      batch_index: the index of the batch amongst all batches in the run.
      counters: a dictionary holding 'success' and 'skipped' fields which can
        be updated to keep track of number of successful and failed runs,
        respectively.  If these fields are not updated, then the success/skipped
        counter values shown at the end of evaluation will be incorrect.
      losses_dict: Optional dictonary of scalar loss tensors.

    Returns:
      result_dict: a dictionary of numpy arrays
      result_losses_dict: a dictionary of scalar losses. This is empty if input
        losses_dict is None.
        :param tensor_dict:
        :param sess:
        :param batch_index:
        :param counters:
        :param losses_dict:
        :return:
    """
      # print(eval_config)
      try:
          if not losses_dict:
              losses_dict = {}
          result_dict, result_losses_dict = sess.run( [tensor_dict, losses_dict] )
          counters['success'] += 1
      except tf.errors.InvalidArgumentError:
          logging.info( 'Skipping image' )
          counters['skipped'] += 1
          return {}, {}
      with sess.graph.as_default():
          # global_step = tf.train.global_step( sess, 200000)
          global_step = train_steps
      if batch_index < eval_config.num_visualizations:
          tag = 'image-{}'.format( batch_index )
          eval_util.visualize_detection_results(
              result_dict,
              tag,
              global_step,
              categories=categories,
              summary_dir=eval_dir,
              export_dir=eval_config.visualization_export_dir,
              show_groundtruth=eval_config.visualize_groundtruth_boxes,
              groundtruth_box_visualization_color=eval_config.
                  groundtruth_box_visualization_color,
              min_score_thresh=eval_config.min_score_threshold,
              max_num_predictions=eval_config.max_num_boxes_to_visualize,
              skip_scores=eval_config.skip_scores,
              skip_labels=eval_config.skip_labels,
              keep_image_id_for_visualization_export=eval_config.
                  keep_image_id_for_visualization_export )
      return result_dict, result_losses_dict

  saver = tf.train.Saver( variables_to_restore )

  def _restore_latest_checkpoint(sess):
      latest_checkpoint = tf.train.latest_checkpoint( checkpoint_dir )
      # print(latest_checkpoint)
      saver.restore( sess, latest_checkpoint )
      # with sess.graph.as_default():
      #     print([n.name for n in tf.get_default_graph().as_graph_def().node])

  metrics = _run_checkpoint_once( tensor_dict,
                                  evaluators=evaluator_list,
                                  batch_processor=_process_batch,
                                  checkpoint_dirs=[checkpoint_dir],
                                  variables_to_restore=None,
                                  restore_fn=_restore_latest_checkpoint,
                                  num_batches=eval_config.num_examples,
                                  master='',
                                  save_graph=False,
                                  save_graph_dir='',
                                  losses_dict=None )

  if not os.path.exists(eval_dir):
      os.makedirs(eval_dir)
  with open(os.path.join(eval_dir, "evaluation_results.json"), 'w') as f:
      f.write(json.dumps(metrics))
  tf.reset_default_graph()
  return metrics


def first_stage_traing(config, input_config):
  print("First stage traing:")
  model_config, train_config, eval_config = config.values()
  models = {}

  for i in range(gen_iter):
    model_idx = "1_{}".format(i)
    upd_configs, params = update_config(config)
    upd_configs = update_augmentation_options(upd_configs)
    model_config, train_config, eval_config = upd_configs.values()
    
    model = Model(upd_configs, params)

    train_dir = os.path.join(FLAGS.train_dir, model_idx)
    if not os.path.exists(train_dir):
      os.makedirs(train_dir)
    
    total_loss = train_process(model_config, input_config, train_config, train_dir)

    # def evaluate(config, create_input_dict_fn, model_fn, train_dir, eval_dir):
    eval_dir = train_dir.replace('train', 'eval')
    evaluation = evaluate([model_config, eval_config, input_config], train_dir, eval_dir)
    model.set_total_loss(total_loss)
    model.set_eval_metrics(evaluation)

    models[model_idx] = model


  # print({i: history[i]['total_loss'] for i in history})
  # print(history)

  best_models = sorted(models, key=lambda k: models[k].total_loss, reverse=True)[:top]

  print('Best models: %s' % best_models)
  best_models = {i: models[i] for i in best_models}
  print('Best models: %s' % best_models)
  return best_models

def second_stage_traing(models, configs, input_config):
  orig_models = models.copy()
  for i in range(top):
    model_id = "2_1_{}".format(i)
    upd_configs, params = shuffle_params(configs, orig_models)

    model_config, train_config, eval_config = upd_configs.values()
    
    model = Model(upd_configs, params)

    train_dir = os.path.join(FLAGS.train_dir, model_id)
    if not os.path.exists(train_dir):
      os.makedirs(train_dir)

    total_loss = train_process(model_config, input_config, train_config, train_dir)

    # def evaluate(config, create_input_dict_fn, model_fn, train_dir, eval_dir):
    eval_dir = train_dir.replace('train', 'eval')
    evaluation = evaluate([model_config, eval_config, input_config], train_dir, eval_dir)
    model.set_total_loss(total_loss)
    model.set_eval_metrics(evaluation)

    models[model_id] = model
  print(["{}: {}".format(i, models[i].total_loss) for i in models])

  for i in range(2):
    model_id = "2_2_{}".format(i)
    upd_configs, params = update_config(configs)
    upd_configs = update_augmentation_options(upd_configs)
    model_config, train_config, eval_config = upd_configs.values()
    
    model = Model(upd_configs, params)

    train_dir = os.path.join(FLAGS.train_dir, model_id)
    if not os.path.exists(train_dir):
      os.makedirs(train_dir)
    
    total_loss = train_process(model_config, input_config, train_config, train_dir)

    # def evaluate(config, create_input_dict_fn, model_fn, train_dir, eval_dir):
    eval_dir = train_dir.replace('train', 'eval')
    evaluation = evaluate([model_config, eval_config, input_config], train_dir, eval_dir)
    model.set_total_loss(total_loss)
    model.set_eval_metrics(evaluation)
    models[model_id] = model
  return models




def main(_):
  assert FLAGS.train_dir, '`train_dir` is missing.'
  if FLAGS.pipeline_config_path:
    model_config, train_config, eval_config, input_config = get_configs_from_pipeline_file(FLAGS.pipeline_config_path)
  else:
    model_config, train_config, input_config = get_configs_from_multiple_files()

  train_config.num_steps = train_steps
  configs = {
              'model': model_config,
              'train_config': train_config,
              'eval_config': eval_config
            }

  # models = first_stage_traing(configs, input_config)

  print("Second stage training:")
  print("Building new model with shuffled best models params")
  # with open('first_stage_best_models.pkl', 'wb') as f:
    # pickle.dump(models, f)

  with open('first_stage_best_models.pkl', 'rb') as f:
    models = pickle.load(f)

  print(models)


  models = second_stage_traing(models, configs, input_config)
  print(models)
  with open('second_stage_traing.pkl', 'wb') as f:
    pickle.dump(models, f)
  




if __name__ == '__main__':
  tf.app.run()

