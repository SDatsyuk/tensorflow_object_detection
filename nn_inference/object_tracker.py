# import the necessary packages
from centroid_original import CentroidTracker
# from centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

import tensorflow as tf
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pbtxt", required=True,
    help="path to Tensorflow 'deploy' prototxt file")
ap.add_argument("-n", "--number_of_classes", type=int, required=True, help="Number of classes")
ap.add_argument("-m", "--model", required=True,
    help="path to Tensorflow pre-trained model")
ap.add_argument("-c", "--cutoff", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

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
    # print(output_dict['num_detections'])
    # print(output_dict['detection_classes'])
    # print(output_dict['detection_scores'])
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)
 
# load our serialized model from disk
print("[INFO] loading model...")
# net = cv2.dnn.readNetFromTensorflow(args["model"], args["pbtxt"])
detection_graph = tf.Graph()
with detection_graph.as_default():
    # with sess.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(args['model'], 'rb') as fid:
       serialized_graph = fid.read()
       od_graph_def.ParseFromString(serialized_graph)
       tf.import_graph_def(od_graph_def, name='')

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2

sess =  tf.Session(graph=detection_graph, config=config)

label_map = label_map_util.load_labelmap(args['pbtxt'])
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=args['number_of_classes'], use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Read and preprocess an image.
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0+cv2.CAP_DSHOW)
vs.set(cv2.CAP_PROP_SETTINGS, 1)
while True:
    _, frame = vs.read()
    rows = frame.shape[0]
    cols = frame.shape[1]
    # frame = cv2.resize(frame, (300, 300))
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # frame = frame[...,[2,0,1]]
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Run the model
    output_dict = run_inference_for_single_image(frame, sess.graph)
    # print(output_dict['detection_classes'][0], output_dict['detection_scores'][0])

    # print(output_dict)
    vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            max_boxes_to_draw=None,
            min_score_thresh = args['cutoff'],
            use_normalized_coordinates=True,
            line_thickness=7)
        # Visualize detected bounding boxes.
 
# initialize the video stream and allow the camera sensor to warmup
# vs.set(cv2.CAP_PROP_SETTINGS, 1)

    # loop over the frames from the video stream
    # loop over the detections
    rects = []
    # print(output_dict["detection_scores"].shape[0])
    for i in range(0, output_dict["detection_scores"].shape[0]):
    #     # filter out weak out by ensuring the predicted
    #     # probability is greater than a minimum threshold
        if output_dict["detection_scores"][i] > args["cutoff"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object, then update the bounding box rectangles list
            box = (output_dict['detection_boxes'][i] * np.array([H, W, H, W])).astype("int"), category_index[output_dict['detection_classes'][i]]['name']
            rects.append(box)
 
            # draw a bounding box surrounding the object so we can
            # visualize it
            # (startX, startY, endX, endY) = box.astype("int")
            # cv2.rectangle(frame, (startX, startY), (endX, endY),
                # (0, 255, 0), 2)
    # # update our centroid tracker using the computed set of bounding
    # # box rectangles
    # print(rects)
    objects = ct.update(rects)
 
    # # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
     
    # #     # show the output frame
    cv2.imshow('TensorFlow MobileNet-SSD', frame)
    key = cv2.waitKey(1) & 0xFF
     
        # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
 
# do a bit of cleanup
print(ct.detected_objects)
cv2.destroyAllWindows()
vs.release()