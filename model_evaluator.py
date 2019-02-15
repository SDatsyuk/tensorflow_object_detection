# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

r"""Evaluation executable for detection models.

This executable is used to evaluate DetectionModels. There are two ways of
configuring the eval job.

1) A single pipeline_pb2.TrainEvalPipelineConfig file maybe specified instead.
In this mode, the --eval_training_data flag may be given to force the pipeline
to evaluate on training data instead.

Example usage:
    ./eval \
        --logtostderr \
        --checkpoint_dir=path/to/checkpoint_dir \
        --eval_dir=path/to/eval_dir \
        --pipeline_config_path=pipeline_config.pbtxt

2) Three configuration files may be provided: a model_pb2.DetectionModel
configuration file to define what type of DetectionModel is being evaulated, an
input_reader_pb2.InputReader file to specify what data the model is evaluating
and an eval_pb2.EvalConfig file to configure evaluation parameters.

Example usage:
    ./eval \
        --logtostderr \
        --checkpoint_dir=path/to/checkpoint_dir \
        --eval_dir=path/to/eval_dir \
        --eval_config_path=eval_config.pbtxt \
        --model_config_path=model_config.pbtxt \
        --input_config_path=eval_input_config.pbtxt
"""
import functools
import tensorflow as tf

from google.protobuf import text_format
from tensorflow import Session

from tensorflow.python.framework import ops

from object_detection.builders import input_reader_builder
from object_detection.builders import model_builder
from object_detection.protos import eval_pb2
from object_detection.protos import input_reader_pb2
from object_detection.protos import model_pb2
from object_detection.protos import pipeline_pb2
from object_detection.utils import label_map_util
from object_detection.utils import object_detection_evaluation
from object_detection import eval_util
from object_detection.metrics import coco_evaluation

from object_detection import evaluator

import os
import numpy as np
import collections
import logging
import json

EVAL_DEFAULT_METRIC = 'pascal_voc_detection_metrics'

EVAL_METRICS_CLASS_DICT = {
      'pascal_voc_detection_metrics':
          object_detection_evaluation.PascalDetectionEvaluator,
      'weighted_pascal_voc_detection_metrics':
          object_detection_evaluation.WeightedPascalDetectionEvaluator,
      'pascal_voc_instance_segmentation_metrics':
          object_detection_evaluation.PascalInstanceSegmentationEvaluator,
      'weighted_pascal_voc_instance_segmentation_metrics':
          object_detection_evaluation.WeightedPascalInstanceSegmentationEvaluator,
      'open_images_V2_detection_metrics':
          object_detection_evaluation.OpenImagesDetectionEvaluator,
      'coco_detection_metrics':
          coco_evaluation.CocoDetectionEvaluator,
      'coco_mask_metrics':
          coco_evaluation.CocoMaskEvaluator,
      'oid_challenge_object_detection_metrics':
          object_detection_evaluation.OpenImagesDetectionChallengeEvaluator,
  }

def _run_checkpoint_once(tensor_dict,
                         evaluators=None,
                         batch_processor=None,
                         checkpoint_dirs=None,
                         variables_to_restore=None,
                         restore_fn=None,
                         num_batches=1,
                         master='',
                         save_graph=False,
                         save_graph_dir='',
                         losses_dict=None):
    """Evaluates metrics defined in evaluators and returns summaries.

  This function loads the latest checkpoint in checkpoint_dirs and evaluates
  all metrics defined in evaluators. The metrics are processed in batch by the
  batch_processor.

  Args:
    tensor_dict: a dictionary holding tensors representing a batch of detections
      and corresponding groundtruth annotations.
    evaluators: a list of object of type DetectionEvaluator to be used for
      evaluation. Note that the metric names produced by different evaluators
      must be unique.
    batch_processor: a function taking four arguments:
      1. tensor_dict: the same tensor_dict that is passed in as the first
        argument to this function.
      2. sess: a tensorflow session
      3. batch_index: an integer representing the index of the batch amongst
        all batches
      By default, batch_processor is None, which defaults to running:
        return sess.run(tensor_dict)
      To skip an image, it suffices to return an empty dictionary in place of
      result_dict.
    checkpoint_dirs: list of directories to load into an EnsembleModel. If it
      has only one directory, EnsembleModel will not be used --
        a DetectionModel
      will be instantiated directly. Not used if restore_fn is set.
    variables_to_restore: None, or a dictionary mapping variable names found in
      a checkpoint to model variables. The dictionary would normally be
      generated by creating a tf.train.ExponentialMovingAverage object and
      calling its variables_to_restore() method. Not used if restore_fn is set.
    restore_fn: None, or a function that takes a tf.Session object and correctly
      restores all necessary variables from the correct checkpoint file. If
      None, attempts to restore from the first directory in checkpoint_dirs.
    num_batches: the number of batches to use for evaluation.
    master: the location of the Tensorflow session.
    save_graph: whether or not the Tensorflow graph is stored as a pbtxt file.
    save_graph_dir: where to store the Tensorflow graph on disk. If save_graph
      is True this must be non-empty.
    losses_dict: optional dictionary of scalar detection losses.

  Returns:
    global_step: the count of global steps.
    all_evaluator_metrics: A dictionary containing metric names and values.

  Raises:
    ValueError: if restore_fn is None and checkpoint_dirs doesn't have at least
      one element.
    ValueError: if save_graph is True and save_graph_dir is not defined.
  """

    global result_losses_dict
    if save_graph and not save_graph_dir:
        raise ValueError( '`save_graph_dir` must be defined.' )
    sess: Session = tf.Session( master, graph=tf.get_default_graph() )
    sess.run( tf.global_variables_initializer() )
    sess.run( tf.local_variables_initializer() )
    sess.run( tf.tables_initializer() )
    if restore_fn:
        restore_fn( sess )
    else:
        if not checkpoint_dirs:
            raise ValueError( '`checkpoint_dirs` must have at least one entry.' )
        checkpoint_file = tf.train.latest_checkpoint( checkpoint_dirs[0] )
        saver = tf.train.Saver( variables_to_restore )
        saver.restore( sess, checkpoint_file )

    if save_graph:
        tf.train.write_graph( sess.graph_def, save_graph_dir, 'eval.pbtxt' )

    counters = {'skipped': 0, 'success': 0}
    aggregate_result_losses_dict = collections.defaultdict( list )
    with tf.contrib.slim.queues.QueueRunners( sess ):
        try:
            for batch in range( int( num_batches ) ):
                if (batch + 1) % 100 == 0:
                    logging.info( 'Running eval ops batch %d/%d', batch + 1, num_batches )
                if not batch_processor:
                    try:
                        if not losses_dict:
                            losses_dict = {}
                        result_dict, result_losses_dict = sess.run( [tensor_dict,
                                                                     losses_dict] )
                        counters['success'] += 1
                    except tf.errors.InvalidArgumentError:
                        logging.info( 'Skipping image' )
                        counters['skipped'] += 1
                        result_dict = {}
                else:
                    result_dict, result_losses_dict = batch_processor(
                        tensor_dict, sess, batch, counters, losses_dict=losses_dict )
                if not result_dict:
                    continue
                for key, value in iter( result_losses_dict.items() ):
                    aggregate_result_losses_dict[key].append( value )
                for evaluator in evaluators:
                    # TODO(b/65130867): Use image_id tensor once we fix the input data
                    # decoders to return correct image_id.
                    # TODO(akuznetsa): result_dict contains batches of images, while
                    # add_single_ground_truth_image_info expects a single image. Fix
                    evaluator.add_single_ground_truth_image_info(
                        image_id=batch, groundtruth_dict=result_dict )
                    evaluator.add_single_detected_image_info(
                        image_id=batch, detections_dict=result_dict )
            logging.info( 'Running eval batches done.' )
        except tf.errors.OutOfRangeError:
            logging.info( 'Done evaluating -- epoch limit reached' )
        finally:
            # When done, ask the threads to stop.
            logging.info( '# success: %d', counters['success'] )
            logging.info( '# skipped: %d', counters['skipped'] )
            all_evaluator_metrics = {}
            for evaluator in evaluators:
                metrics = evaluator.evaluate()
                evaluator.clear()
                if any( key in all_evaluator_metrics for key in metrics ):
                    raise ValueError( 'Metric names between evaluators must not collide.' )
                all_evaluator_metrics.update( metrics )

            with sess.graph.as_default():
                # global_step = tf.train.global_step( sess, tf.train.get_global_step() )
                global_step = 200000
            for key, value in iter( aggregate_result_losses_dict.items() ):
                all_evaluator_metrics['Losses/' + key] = np.mean( value )
    sess.close()
    # print( all_evaluator_metrics )
    return all_evaluator_metrics


def get_configs_from_pipeline_file():
    """Reads evaluation configuration from a pipeline_pb2.TrainEvalPipelineConfig.

  Reads evaluation config from file specified by pipeline_config_path flag.

  Returns:
    model_config: a model_pb2.DetectionModel
    eval_config: a eval_pb2.EvalConfig
    input_config: a input_reader_pb2.InputReader
  """
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile( FLAGS.pipeline_config_path, 'r' ) as f:
        text_format.Merge( f.read(), pipeline_config )

    model_config = pipeline_config.model
    if FLAGS.eval_training_data:
        eval_config = pipeline_config.train_config
    else:
        eval_config = pipeline_config.eval_config
    input_config = pipeline_config.eval_input_reader

    return model_config, eval_config, input_config


def get_evaluators(eval_config, categories, metrics=EVAL_DEFAULT_METRIC):
    """Returns the evaluator class according to eval_config, valid for categories.

  Args:
    eval_config: evaluation configurations.
    categories: a list of categories to evaluate.
  Returns:
    An list of instances of DetectionEvaluator.

  Raises:
    ValueError: if metric is not in the metric class dictionary.
  """
    eval_metric_fn_keys = eval_config.metrics_set
    if not eval_metric_fn_keys:
        eval_metric_fn_keys = [metrics]
    evaluators_list = []
    for eval_metric_fn_key in eval_metric_fn_keys:
        if eval_metric_fn_key not in metrics:
            raise ValueError( 'Metric not found: {}'.format( eval_metric_fn_key ) )
        evaluators_list.append(
            EVAL_METRICS_CLASS_DICT[eval_metric_fn_key]( categories=categories ) )
    return evaluators_list


def get_configs_from_multiple_files():
    """Reads evaluation configuration from multiple config files.

  Reads the evaluation config from the following files:
    model_config: Read from --model_config_path
    eval_config: Read from --eval_config_path
    input_config: Read from --input_config_path

  Returns:
    model_config: a model_pb2.DetectionModel
    eval_config: a eval_pb2.EvalConfig
    input_config: a input_reader_pb2.InputReader
  """
    eval_config = eval_pb2.EvalConfig()
    with tf.gfile.GFile( FLAGS.eval_config_path, 'r' ) as f:
        text_format.Merge( f.read(), eval_config )

    model_config = model_pb2.DetectionModel()
    with tf.gfile.GFile( FLAGS.model_config_path, 'r' ) as f:
        text_format.Merge( f.read(), model_config )

    input_config = input_reader_pb2.InputReader()
    with tf.gfile.GFile( FLAGS.input_config_path, 'r' ) as f:
        text_format.Merge( f.read(), input_config )

    return model_config, eval_config, input_config


def main(unused_argv):
    assert FLAGS.checkpoint_dir, '`checkpoint_dir` is missing.'
    assert FLAGS.eval_dir, '`eval_dir` is missing.'
    if FLAGS.pipeline_config_path:
        model_config, eval_config, input_config = get_configs_from_pipeline_file()
    else:
        model_config, eval_config, input_config = get_configs_from_multiple_files()

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

    evaluator_list = get_evaluators( eval_config, categories )

    checkpoint_dir = FLAGS.checkpoint_dir
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
            global_step = 200000
        if batch_index < eval_config.num_visualizations:
            tag = 'image-{}'.format( batch_index )
            eval_util.visualize_detection_results(
                result_dict,
                tag,
                global_step,
                categories=categories,
                summary_dir=FLAGS.eval_dir,
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
                                    checkpoint_dirs=[FLAGS.checkpoint_dir],
                                    variables_to_restore=None,
                                    restore_fn=_restore_latest_checkpoint,
                                    num_batches=eval_config.num_examples,
                                    master='',
                                    save_graph=False,
                                    save_graph_dir='',
                                    losses_dict=None )

    if not os.path.exists(FLAGS.eval_dir):
        os.makedirs(FLAGS.eval_dir)
    with open(os.path.join(FLAGS.eval_dir, "evaluation_results.json"), 'w') as f:
        f.write(json.dumps(metrics))


if __name__ == '__main__':

  if not os.environ["CUDA_VISIBLE_DEVICES"]:
      os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  logging.basicConfig(level=logging.INFO)

  

  tf.logging.set_verbosity( tf.logging.INFO )

  flags = tf.app.flags
  flags.DEFINE_boolean('eval_training_data', False,
                        'If training data should be evaluated for this job.')
  flags.DEFINE_string( 'checkpoint_dir', '',
                       'Directory containing checkpoints to evaluate, typically '
                       'set to `train_dir` used in the training job.' )
  flags.DEFINE_string( 'eval_dir', '',
                       'Directory to write eval summaries to.' )
  flags.DEFINE_string( 'pipeline_config_path', '',
                       'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                       'file. If provided, other configs are ignored' )
  flags.DEFINE_string( 'eval_config_path', '',
                       'Path to an eval_pb2.EvalConfig config file.' )
  flags.DEFINE_string( 'input_config_path', '',
                       'Path to an input_reader_pb2.InputReader config file.' )
  flags.DEFINE_string( 'model_config_path', '',
                       'Path to a model_pb2.DetectionModel config file.' )

  FLAGS = flags.FLAGS

  tf.app.run()  
