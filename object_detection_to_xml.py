
# coding: utf-8

# # Object Detection Demo
"""
  Usage:
  python object_detection_custom.py -c philmor49.pb -l maps/philmor49.pbtxt -n 49 -i test_images -a annotations
"""


# # Imports
import numpy as np
import os
import cv2
import json
import time

import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
# import matplotlib; matplotlib.use('Agg')
#import matplotlib
#matplotlib.use("Pdf")
from matplotlib import pyplot as plt
from PIL import Image

import glob
# import xml.etree.ElementTree as ET
from lxml import etree as ET

from PIL import Image


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

from xml_utils import *

# if tf.__version__ < '1.4.0':
#     raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from utils import label_map_util

from utils import visualization_utils as vis_util

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoint", type=str, help='path to checkpoint')
ap.add_argument('-l', '--labels', type=str, help='Path to labels (*.pbtxt)')
ap.add_argument('-n', '--num_classes', type=int, help='Number of classes')
ap.add_argument("-i", "--input_dir", type=str, help='input directory')
ap.add_argument('-s', '--save', type=str, default='y', help='Paste `y` to save output images')
ap.add_argument("-o", "--output_dir", type=str, help="output directory")
ap.add_argument('-t', '--threshold', type=float, default=.5, help='boxes threshold')
ap.add_argument('-a', "--annotation_dir", default=None, type=str, help="Annotation directory")
args = vars(ap.parse_args())

print(args)

SVM_MODEL = None
DT_MODEL = None

detection_graph = tf.Graph()
with detection_graph.as_default():
    # with sess.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(args['checkpoint'], 'rb') as fid:
       serialized_graph = fid.read()
       od_graph_def.ParseFromString(serialized_graph)
       tf.import_graph_def(od_graph_def, name='')

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2

sess =  tf.Session(graph=detection_graph, config=config)

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(args['labels'])
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=args['num_classes'], use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.shape[:2]
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = 'test_images'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(3) ]
TEST_IMAGE_PATHS = glob.glob1(args['input_dir'], '*.jpg')

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 16)
print(TEST_IMAGE_PATHS)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        # with tf.Session() as sess:
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
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
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
      #print(output_dict['detection_scores'])
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


# non-maximum suppression function
def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return np.array([]).astype("int")

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    sc = boxes[:,4]
    cl = boxes[:,5]
 
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(sc)
 
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        #todo fix overlap-contains...
        overlap = (w * h) / area[idxs[:last]]
         
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    
    return boxes[pick].astype("int")

def decode_prediction(preds, im_width, im_height):
    detections = [(normalize_coord(box, im_width, im_height), score, label) for box, label, score in zip(preds['detection_boxes'], preds['detection_classes'], preds['detection_scores']) if score > .2]
    # for i in range(100):
    #   if preds[]
    return detections

def normalize_coord(box, im_width, im_height):
    ymin, xmin, ymax, xmax = box.tolist()
    left, right, top, bottom = [int(i) for i in (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)]
    return left, right, top, bottom

def draw_boxes(image, boxes):
    for x0, y0, x1, y1, score, label in boxes:
        image = cv2.rectangle(image, (x0, y0), (x1, y1), (0,255,0), 5)
    return image


for image_path in TEST_IMAGE_PATHS:
    start = time.time()
    print("Reading image %s" % image_path)
    # PILLOW
    # image = Image.open(os.path.join(args['input_dir'], image_path))
    # opencv
    image = cv2.imread(os.path.join(args['input_dir'], image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_height, im_width = image.shape[:2]
    # print(image.shape[:2])
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    # image_np = load_image_into_numpy_array(image)
    image_np = image
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    print("Preproc time: {}".format(time.time() - start))
    start = time.time()
    output_dict = run_inference_for_single_image(image_np, sess.graph)
    print("Rec time: {}".format(time.time()-start))
  # Visualization of the results of a detection.

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        max_boxes_to_draw=None,
        min_score_thresh=args['threshold'],
        line_thickness=8)

    if args['save'] == 'y':
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        if not os.path.isdir(args['output_dir']):
            os.makedirs(args['output_dir'])
        plt.savefig(os.path.join(args['output_dir'], image_path))
        plt.close("all")
        # plt.show()

    boxes = []
    # # print(output_dict['detection_scores'])
    # # break
    for i in range(100):
        # print(output_dict['detection_scores'])
        if output_dict['detection_scores'][i] < args['threshold']:
            break
        y0, x0, y1, x1, score, label = *output_dict['detection_boxes'][i], output_dict['detection_scores'][i], output_dict['detection_classes'][i]
        y0, x0, y1, x1, score, label = int(x0 * im_width), int(y0 * im_height), int(x1 * im_width), int(y1 * im_height), score, label
        boxes.append((y0, x0, y1, x1, score, label))

    # # print(boxes)
    # # print("Non maximum suppresion")
    # boxes = non_max_suppression(np.array(boxes), 0.4).tolist()

    #   # boxes = sorted(sorted(boxes, key=lambda k: k[4]), key=lambda k: k[0])
    # boxes = sorted(sorted(boxes, key=lambda k: k[1]), key=lambda k: k[0])
    # print(boxes)
    # # print("XML main")

    # create main xml block
    if args['annotation_dir']:

        annotation = create_xml_main(folder_text=args['input_dir'], filename_text=image_path, image_shape=image.shape[:2])

        # print("Creating boxes")
        # create object xml tag for each recognized object
        for i in boxes:
            pack_class = category_index[int(i[5])]['name']
                
            object_tag = create_annotation_object(pack_name=pack_class, xmin=i[0], ymin=i[1], xmax=i[2], ymax=i[3], pose='Unspecified', truncated='0', difficult='0')
            annotation.extend(object_tag)

        # print(ET.dump(annotation))
        # convert ET object to string
        annotation_str = ET.tostring(annotation, pretty_print=True)

        # save xml file to `annotation_dir`
        s = save_xml(annotation_str, args['annotation_dir'], image_path[:-4] + ".xml")
        print("Saving: %s" % s)
  # break

  # image = draw_boxes(image_np, boxes)
  # plt.figure(figsize=(30, 30))
  # plt.imshow(image)
  # plt.savefig(os.path.join('test_images', 'rec_res', image_path))

  
  # print(output_dict['detection_boxes'])
  # print(output_dict['detection_classes'])
  # print(output_dict['detection_scores'])
  #np.save("/home/pi/res.npy", output_dict)
  #print('Saving image')
  #cv2.imwrite('/home/pi/res_{}.jpg'.format(image_path), image_np)
  # break
