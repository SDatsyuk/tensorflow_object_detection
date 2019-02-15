import tensorflow as tf
import os
import argparse
import numpy as np
import cv2
from collections import deque
import time
import random

import label_map_util
import visualization_utils as vis_util

from utils import *
from centroidtracker import CentroidTracker
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

ap = argparse.ArgumentParser()
ap.add_argument("--input", default=0, help="video input or camera index [default=0] ")
ap.add_argument('--model', required=True, help='path to model file')
ap.add_argument("--labels", required=True, help='path to label pbtxt file')
ap.add_argument("--num_classes", required=True, help='number of classes')
ap.add_argument("--tracking", action="store_true", help="apply tracking for recognized objects")
ap.add_argument("--direction", action="store_true", help="apply direction for recognized objects")
ap.add_argument("-t", "--threshold", type=float, default=0.5)
# ap.add_argument("-b", "--direction_buffer", type=int, default=32,
    # help="max direction buffer size")

args = vars(ap.parse_args())
# pts = deque(maxlen=/args['direction_buffer'])
direction = ""

def probnet():
    # Defining the model structure. We can define the network by just passing a list of edges.
    model = BayesianModel([('H', 'S'), ('B', 'S'), ('D', 'S')])
    # Defining individual CPDs.
    cpd_h = TabularCPD(variable='H', variable_card=2, values=[[0.2, 0.8]])
    cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.1, 0.9]])
    cpd_d = TabularCPD(variable='D', variable_card=2, values=[[0.5, 0.5]])
    cpd_s = TabularCPD(variable='S', variable_card=2, 
                       values=[[0.1, 0.2, 0.1, 0.15, 0.4, 0.35, 0.45, 0.43],
                               [0.9, 0.8, 0.9, 0.85, 0.6, 0.65, 0.55, 0.57]],
                      evidence=['H', 'B', 'D'],
                      evidence_card=[2, 2, 2])
    # Associating the CPDs with the network
    model.add_cpds(cpd_h, cpd_b, cpd_d, cpd_s)
    # check_model checks for the network structure and CPDs and verifies that the CPDs are correctly 
    # defined and sum to 1.
    model.check_model()
    print(model.get_cpds('S'))
    # infer = VariableElimination(model)
    # infer.map_query('S', evidence={'H': 1, 'B': 0, 'D': 1})
    return model

def probnet_inference(model, h, b, d):
    H = 1 if h > 10 else 0
    B = 1 if b > 20 else 0
    D = 1 if d > 3 else 0
    print(H, B, D)
    infer = VariableElimination(model)
    return infer.map_query(['S'], evidence={'H': H, 'B': B, 'D': D})


def inference():

    # prob_model = probnet()

    counter = 0
    sess = tf.Session()
    detection_graph = load_model(args['model'], sess).graph
    labels = load_labels(args['labels'], int(args['num_classes']))

    cap = cv2.VideoCapture(int(args['input'])+cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_SETTINGS, 1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    h, w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH )
    print(h, w)

    if args['tracking']:
        ct = CentroidTracker()
        (H, W) = (None, None)
        disapear_delay = 2 # seconds
        maxDisappeared = None
    
    wrong_direction = 0


    while True:

        start = time.time()

        _, frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        start = time.time()
        output_dict = run_inference_for_single_image(frame, detection_graph, sess)
        print('Inference time: %s' % (time.time() - start))

        if args['direction']:
            obj = [output_dict['detection_boxes'][i] for i in range(output_dict['num_detections']) if output_dict['detection_scores'][i] > args['threshold']]
            
            # if obj:
        #         obj = obj[0]
        #         center = (int((obj[3] - (obj[3] - obj[1]) / 2) * w) ,int((obj[2] - (obj[2] - obj[0]) / 2) * h))
        #             # print(objects)
        #         pts.appendleft(center)

        #         counter = find_direction(frame, pts, counter, args['direction_buffer'])

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

        if args['tracking']:
            rects = []
            # print(output_dict["detection_scores"].shape[0])
            for i in range(0, output_dict["detection_scores"].shape[0]):
                # filter out weak out by ensuring the predicted
                # probability is greater than a minimum threshold
                if output_dict["detection_scores"][i] < args["threshold"]:
                    break
                    # compute the (x, y)-coordinates of the bounding box for
                    # the object, then update the bounding box rectangles list
                box = (output_dict['detection_boxes'][i] * np.array([H, W, H, W])).astype("int"), labels[output_dict['detection_classes'][i]]['name']
                rects.append(box)

            objects = ct.update(rects)
            
     
            # loop over the tracked objects
            y_pad = 20
            for (objectID, centroid) in objects.items():
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                direction = "ID {}: {}".format(objectID, ct.objectDirection[objectID])

                cv2.putText(frame, direction, (10, y_pad), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y_pad += 15

                if ct.preds[objectID] == 'hand_with_somethinh' and ct.objectDirection[objectID] == "to basket":
                    wrong_direction += 1

                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('reco', frame)

        end = time.time()
        ct.maxDisappeared = int(disapear_delay / (end - start))
        # print(ct.maxDisappeared)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
    print("Hand disapeared %s times" % ct.deregistered)
    print("Hand moves with product to the basket %s times" % wrong_direction)

    prob_score = probnet_inference(prob_model, ct.deregistered, wrong_direction, random.randint(1, 10))
    print("Probability model result: %s" % prob_score)

if __name__ == "__main__":
    inference()
    # prob_model = probnet()
    # prob_score = probnet_inference(prob_model, random.randint(0, 10), random.randint(1, 8), random.randint(1, 10))
    # print("Probability model result: %s" % prob_score)