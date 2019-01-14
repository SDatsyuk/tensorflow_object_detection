"""
Usage:
    python3 view_record.py --record=data.record
"""
import argparse

import numpy as np
import tensorflow as tf

import cv2
import sys

FLAGS = None


def _extract_feature(element):
    """
    Extract features from a single example from dataset.
    """
    features = tf.parse_single_example(
        element,
        # Defaults are not specified since both keys are required.
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(tf.float32)
        })
    return features


def show_record(filenames):
    """.
    Show the TFRecord contents
    """
    # Generate dataset from TFRecord file.
    dataset = tf.data.TFRecordDataset(filenames)

    # Make dataset iteratable.
    iterator = dataset.make_one_shot_iterator()
    print(iterator)
    next_example = iterator.get_next()
    print(next_example)

    # Extract features from single example
    features = _extract_feature(next_example)
    print(features)
    image_decoded = tf.image.decode_image(features['image/encoded'])
    xmin = tf.cast(features['image/object/bbox/xmin'], tf.float32)
    xmax = tf.cast(features['image/object/bbox/xmax'], tf.float32)
    ymin = tf.cast(features['image/object/bbox/ymin'], tf.float32)
    ymax = tf.cast(features['image/object/bbox/ymax'], tf.float32)
    print(image_decoded)
    # sys.exit(0)

    # Use openCV for preview
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    # Actrual session to run the graph.
    with tf.Session() as sess:
        while True:
            try:
                image_tensor, xmin1, ymin1, xmax1, ymax1 = sess.run(
                    [image_decoded, xmin, ymin, xmax, ymax])
                # print(label_text.shape)
                print(xmin1)
                print(xmax1)


                # Use OpenCV to preview the image.
                image = np.array(image_tensor, np.uint8)
                cv2.imshow("image", image)
                cv2.waitKey(0)

                # Show the labels
            except tf.errors.OutOfRangeError:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record",
        type=str,
        default="train.record",
        help="The record file."
    )
    args = parser.parse_args()
    show_record(args.record)