import argparse
import tensorflow as tf
import numpy as np
import cv2
import os

ap = argparse.ArgumentParser()

ap.add_argument('-f', "--file", required=True, help="Tfrecord file path")

args = vars(ap.parse_args())

def check_dir(path):
	if not os.path.exists(path):
		os.mkdir(path)


session = tf.Session()

save_path = 'output/products/parts'

for string_record in tf.python_io.tf_record_iterator(args['file']):
    example = tf.train.Example()
    example.ParseFromString(string_record)
    # print(dir(example.features))
    # print(example)
    
    height = int(example.features.feature['image/height']
                                 .int64_list
                                 .value[0])
    
    width = int(example.features.feature['image/width']
                                .int64_list
                                .value[0])
    
    img_string = (example.features.feature['image/encoded']
                                  .bytes_list
                                  .value[0])

    ymax = example.features.feature['image/object/bbox/ymax'].float_list.value
    xmax = example.features.feature['image/object/bbox/xmax'].float_list.value
    ymin = example.features.feature['image/object/bbox/ymin'].float_list.value
    xmin = example.features.feature['image/object/bbox/xmin'].float_list.value
    labels = example.features.feature['image/object/class/text'].bytes_list.value
    image_filename = example.features.feature['image/filename'].bytes_list.value[0]
     
    # annotation_string = (example.features.feature['image/mask_raw']
    #                             .bytes_list
    #                             .value[0])
    image_decoded = tf.image.decode_image(img_string)
    with tf.Session() as sess:
        
        try:
            image_tensor = sess.run(
                [image_decoded])

            # Use OpenCV to preview the image.
            # print(image_tensor)
            image = np.array(image_tensor, np.uint8)
            image = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)
            # print(image.shape)
            # cv2.imshow("image", image)
            # cv2.waitKey(100)
            part_id = 0
            for i, label in enumerate(labels):
            	image_part = image[int(ymin[i] * height):int(ymax[i] * height), int(xmin[i] * width):int(xmax[i] * width)]
            	check_dir(os.path.join(save_path, label.decode()))
            	cv2.imwrite(os.path.join(save_path, label.decode(), "{}_{}.jpg".format(image_filename, part_id)), image_part)
            	part_id += 1
            	# cv2.imshow(label.decode(), image_part)
            	# cv2.waitKey(0)

            # Show the labels
            # print(label_text)
        except tf.errors.OutOfRangeError:
            break
    # break
    # Annotations don't have depth (3rd dimension)
    # reconstructed_annotation = annotation_1d.reshape((height, width))
    
    # reconstructed_images.append((reconstructed_img, reconstructed_annotation))