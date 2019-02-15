import sys
from subprocess import PIPE, STDOUT, Popen, CREATE_NEW_CONSOLE
import time
import argparse
import configparser
import logging
import random
import glob
import os

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from run_utils import create_tf_record, read_config, create_main_dir, list_dir, get_configs_from_pipeline_file

OUTPUT_PATH = "output"

ap = argparse.ArgumentParser()
ap.add_argument("--tfrecord", action="store_true", help="create tfrecord")
ap.add_argument("--train", action="store_true", help="start train process")
ap.add_argument("--eval", action="store_true", help='start eval process')
ap.add_argument("--tensorboard", action="store_true", help='start tensorboard')
ap.add_argument("--export", action="store_true", help='export inference graph from model checkpoint')
ap.add_argument("--config", default="run.config", help="path to .config file")

args = vars(ap.parse_args())


# def create_tfrecord(data_dir, label_map_path, output_dir):
def create_tfrecord(config, output_dir):
	data_dir = config['TFRECORD']['DATA_DIR']
	label_map_dict = label_map_util.get_label_map_dict(config['TFRECORD']['LABELS_PATH'])

	logging.info('Reading from goods dataset.')
	image_dir = os.path.join(data_dir, config['TFRECORD']['IMAGES'])
	annotations_dir = os.path.join(data_dir, config['TFRECORD']['ANNOTATIONS'])
	trainval = config['TFRECORD']['TRAINVAL'] if config['TFRECORD'].get("TRAINVAL") else 'trainval.txt'
	examples_path = list_dir(annotations_dir, os.path.join(output_dir, 'tfrecord', trainval))
	# examples_path = os.path.join(output_dir, 'tfrecord', trainval)
	examples_list = dataset_util.read_examples_list(examples_path)

	# Test images are not included in the downloaded data set, so we shall perform
	# our own split.
	random.seed(42)
	random.shuffle(examples_list)
	num_examples = len(examples_list)
	num_train = int(0.7 * num_examples)
	train_examples = examples_list[:num_train]
	val_examples = examples_list[num_train:]
	logging.info('%d training and %d validation examples.',
	           len(train_examples), len(val_examples))

	train_output_path = os.path.join(output_dir, 'tfrecord', '{}_train.record'.format(config['MODEL']["NAME"]))
	val_output_path = os.path.join(output_dir, 'tfrecord', '{}_eval.record'.format(config['MODEL']["NAME"]))
	create_tf_record(train_output_path, label_map_dict, annotations_dir,
	               image_dir, train_examples)
	create_tf_record(val_output_path, label_map_dict, annotations_dir,
	               image_dir, val_examples)
	return train_output_path, val_output_path

def update_pipeline_config(config, pipeline_config):
	update_list = {}
	if int(config['MODEL']['NUM_CLASSES']):
		if pipeline_config.model.HasField('ssd'):
			pipeline_config.model.ssd.num_classes = int(config["MODEL"]["NUM_CLASSES"])
		elif pipeline_config.model.HasField('faster_rcnn'):
			pipeline_config.model.faster_rcnn.num_classes = int(config["MODEL"]["NUM_CLASSES"])

	if config["MODEL"]['NUM_STEPS']:
		pipeline_config.train_config.num_steps = int(config["MODEL"]["NUM_STEPS"])
	if config["MODEL"]["TRAIN_TFRECORD"]:
		pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] = config['MODEL']['TRAIN_TFRECORD']
	if config["MODEL"]["EVAL_TFRECORD"]:
		pipeline_config.eval_input_reader.tf_record_input_reader.input_path[0] = config["MODEL"]['EVAL_TFRECORD']
	if config["MODEL"]['MAP_FILE']:
		pipeline_config.train_input_reader.label_map_path = config["MODEL"]['MAP_FILE']
		pipeline_config.eval_input_reader.label_map_path = config["MODEL"]['MAP_FILE']
	if config['MODEL']['FINE_TUNE_CHECKPOINT']:
		pipeline_config.train_config.fine_tune_checkpoint = os.path.join(config['MODEL']['FINE_TUNE_CHECKPOINT'], 'model.ckpt')
	pipeline_config.eval_config.visualization_export_dir = os.path.join(main_dir, 'export')
	return pipeline_config


def start_training(train_dir, pipeline_config_path):
	train = Popen('python train.py --logtostderr --train_dir={} --pipeline_config_path={}'.format(train_dir, pipeline_config_path), stdin=None, stdout=None, stderr=None)
	return train

def start_eval(train_dir, eval_dir, pipeline_config_path):
	evl = Popen(r"python eval.py  --logtostderr --checkpoint_dir=path/to/checkpoint_dir --eval_dir=path/to/eval_dir --pipeline_config_path= --logtostderr --checkpoint_dir={} --pipeline_config_path={} --eval_dir={}".format(train_dir, pipeline_config_path, eval_dir), creationflags=CREATE_NEW_CONSOLE,  stderr=None)
	# output, error_output = evl.communicate()
	time.sleep(10)

	# print(error_output)

	return evl

def start_tensorboard(log_dir):
	tensorboard = Popen('tensorboard --logdir={}'.format(log_dir), creationflags=CREATE_NEW_CONSOLE, stdin=None, stdout=None, stderr=None)
	return tensorboard

def export_inference_graph(config):
	train_path = os.path.join(config['MODEL']['WD'], 'train')
	models_list = glob.glob1(train_path, 'model.ckpt-*')
	models_sort = sorted(models_list, key=lambda k: int(k.split('model.ckpt-')[-1].split('.')[0]), reverse=True)
	model = ".".join(models_sort[0].split('.')[:2])
	export = Popen("python export_inference_graph.py  --input_type image_tensor --pipeline_config_path {} --trained_checkpoint_prefix {} --output_directory {}".format(config["MODEL"]["pipeline_config"], os.path.join(train_path, model), os.path.join(config["MODEL"]["WD"], 'pb')))

def main():
	global main_dir
	print(args['config'])
	config = read_config(args['config'])

	print("Creating folders...")
	if not config['MODEL'].get('WD', False):
		main_dir = create_main_dir(config, OUTPUT_PATH)
		config['MODEL']["WD"] = main_dir
	else:
		main_dir = config['MODEL']["WD"]

	if args['tfrecord']:
		print("Creating tfrecord...")
		# tfr_config = config['TFRECORD']
		tfrecord_path = create_tfrecord(config, main_dir)
		if len(tfrecord_path) > 1:
			# print(tfrecord_path)
			config["MODEL"]["TRAIN_TFRECORD"] = tfrecord_path[0]
			config["MODEL"]["EVAL_TFRECORD"] = tfrecord_path[1]
		else:
			config["TRAIN_TFRECORD"] = tfrecord_path

	print("Updating pipeline config...")
	pipeline_config = get_configs_from_pipeline_file(config['MODEL']['PIPELINE_CONFIG'])
	# print(pipeline_config)
	pipeline_config = update_pipeline_config(config, pipeline_config)
	with open(os.path.join(main_dir, config['MODEL']['PIPELINE_CONFIG'].split(os.sep)[-1]), 'w') as f:
		f.write(str(pipeline_config))
		config["MODEL"]['PIPELINE_CONFIG'] = os.path.join(main_dir, config['MODEL']['PIPELINE_CONFIG'].split(os.sep)[-1])
		print(config["MODEL"]['PIPELINE_CONFIG'])
	# print(type(pipeline_config[0]['ssd']))
	# print(pipeline_config.model.HasField('ssd'))

	with open(args['config'], 'w') as f:
		config.write(f)

	
	if args['train']:
		train_process = start_training(os.path.join(main_dir, 'train'), config['MODEL']['PIPELINE_CONFIG'])
		time.sleep(60)

	if args['eval']:
		# print('python eval.py --logtostderr --checkpoint_dir={} --pipeline_config_path={}, --eval_dir={}'.format(args['train_dir'], args['eval_dir'], args['pipeline_config_path']))
		eval_process = start_eval(os.path.join(main_dir, 'train'), os.path.join(main_dir, 'eval'), config["MODEL"]["PIPELINE_CONFIG"])
	if args['tensorboard']:
		tensorboard = start_tensorboard(main_dir)

	if args['export']:
		export_inference_graph(config)


if __name__ == "__main__":
	main()