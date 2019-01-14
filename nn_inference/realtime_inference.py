import tensorflow as tf
import os
import argparse
import numpy as np
import cv2
from collections import deque

import label_map_util
import visualization_utils as vis_util

ap = argparse.ArgumentParser()
ap.add_argument("--input", default=0, help="video input or camera index [default=0] ")
ap.add_argument('--model', required=True, help='path to model file')
ap.add_argument("--labels", required=True, help='path to label pbtxt file')
ap.add_argument("--num_classes", required=True, help='number of classes')
ap.add_argument("-t", "--threshold", default=0.5)
ap.add_argument("-b", "--direction_buffer", type=int, default=32,
	help="max direction buffer size")

args = vars(ap.parse_args())
pts = deque(maxlen=args['direction_buffer'])
counter = 0
direction = ""

def load_model(path, sess):
	
	with sess.graph.as_default():
	  od_graph_def = tf.GraphDef()
	  with tf.gfile.GFile(path, 'rb') as fid:
	    serialized_graph = fid.read()
	    od_graph_def.ParseFromString(serialized_graph)
	    tf.import_graph_def(od_graph_def, name='')
	return sess

def load_labels(path, num_classes):
	label_map = label_map_util.load_labelmap(path)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)
	return category_index

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape(
	  (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph, sess):
	with graph.as_default():
		with sess.as_default():
      # Get handles to input and output tensors
			ops = tf.get_default_graph().get_operations()
			all_tensor_names = {output.name for op in ops for output in op.outputs}
			tensor_dict = {}
			for key in [
				'num_detections', 'detection_boxes', 'detection_scores',
				'detection_classes', 'detection_masks'
			]:
				tensor_name = key + ':0'
				if tensor_name in all_tensor_names:
					tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
			if 'detection_masks' in tensor_dict:
				# The following processing is only for single image
				detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
				detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
				# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
				real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
				detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
				detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
				detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
				detection_masks_reframed = tf.cast(
					tf.greater(detection_masks_reframed, 0.5), tf.uint8)
				# Follow the convention by adding back the batch dimension
				tensor_dict['detection_masks'] = tf.expand_dims(
					detection_masks_reframed, 0)
			image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

			# Run inference
			output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

			# all outputs are float32 numpy arrays, so convert types as appropriate
			output_dict['num_detections'] = int(output_dict['num_detections'][0])
			output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
			output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
			output_dict['detection_scores'] = output_dict['detection_scores'][0]
#      print(output_dict['num_detections'])
#      print(output_dict['detection_classes'])
			# print(output_dict['detection_scores'])
			if 'detection_masks' in output_dict:
				output_dict['detection_masks'] = output_dict['detection_masks'][0]
	return output_dict


def find_direction(frame, pts):
	global counter
	direction = ""
	for i in np.arange(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue
 
		# check to see if enough points have been accumulated in
		# the buffer
		if counter >= 10 and i == 1 and pts[-10] is not None:
			# compute the difference between the x and y
			# coordinates and re-initialize the direction
			# text variables
			dX = pts[-10][0] - pts[i][0]
			dY = pts[-10][1] - pts[i][1]
			(dirX, dirY) = ("", "")
 
			# ensure there is significant movement in the
			# x-direction
			if np.abs(dX) > 20:
				dirX = "to basket" if np.sign(dX) == 1 else "to package"
 
			# ensure there is significant movement in the
			# y-direction
			# if np.abs(dY) > 20:
			# 	dirY = "North" if np.sign(dY) == 1 else "South"
 
			# handle when both directions are non-empty
			direction = dirX
 
			# otherwise, only one direction is non-empty
			# else:
				# direction = dirX if dirX != "" else dirY

		cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (0, 0, 255), 3)
		thickness = int(np.sqrt(args["direction_buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
	counter += 1


def inference():

	sess = tf.Session()
	detection_graph = load_model(args['model'], sess).graph
	labels = load_labels(args['labels'], int(args['num_classes']))

	cap = cv2.VideoCapture(args['input'])

	h, w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH )
	print(h, w)

	while True:
		_, frame = cap.read()

		output_dict = run_inference_for_single_image(frame, detection_graph, sess)

		obj = [output_dict['detection_boxes'][i] for i in range(output_dict['num_detections']) if output_dict['detection_scores'][i] > args['threshold']]
		if obj:
			obj = obj[0]
			# print(obj[2] * w / 2, obj[3] * h / 2)
			# cv2.circle(frame, (int((obj[3] - (obj[3] - obj[1]) / 2) * w) ,int((obj[2] - (obj[2] - obj[0]) / 2) * h)), 5, (0, 0, 255), -1)
			center = (int((obj[3] - (obj[3] - obj[1]) / 2) * w) ,int((obj[2] - (obj[2] - obj[0]) / 2) * h))
				# print(objects)
			pts.appendleft(center)

			find_direction(frame, pts)

		vis_util.visualize_boxes_and_labels_on_image_array(
			frame,
			output_dict['detection_boxes'],
			output_dict['detection_classes'],
			output_dict['detection_scores'],
			labels,
			instance_masks=output_dict.get('detection_masks'),
			use_normalized_coordinates=True,
			line_thickness=8, 
			min_score_thresh=args['threshold']
			)

		cv2.imshow('reco', frame)

		key = cv2.waitKey(1) & 0xFF
		if key == 27:
			break

if __name__ == "__main__":
	inference()