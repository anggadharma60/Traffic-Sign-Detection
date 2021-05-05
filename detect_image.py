import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import time
from threading import Thread
import importlib.util

from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'export/Some/9/150/frozen_inference_graph.pb'


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'data/SomeClass_label.pbtxt'

PATH_TO_IMAGES =  'test/00028_432.jpg'
# PATH_TO_IMAGES =  'test/warning.jpg'
NUM_CLASSES = 4
IMAGE_SIZE = (8,8)

min_conf_threshold = float(0.95)
resW, resH = 640, 480
imW, imH = int(resW), int(resH)
color_box = [(0,255,0), (0,0,255), (0,255,255)]

img = cv2.imread(PATH_TO_IMAGES)
(im_width, im_height, dim) = img.shape
image = np.array(img.reshape((im_height, im_width, dim)).astype(np.uint8))
print(image)
# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print(category_index)
    
        
# # Detection
# with detection_graph.as_default():
#     with tf.compat.v1.Session(graph=detection_graph) as sess:

#         # Grab frame from video stream
#         img = cv2.imread(PATH_TO_IMAGES)
        
#         frame1 = cv2.resize(img,(imW,imH))
        
#         # Acquire frame and resize to expected shape [1xHxWx3]
       
#         frame = frame1.copy()
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         frame_resized = cv2.resize(frame_rgb, (imW, imH))
#         image_np_expanded = np.expand_dims(frame_resized, axis=0)
        
#         # Extract image tensor
#         image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#         # Extract detection boxes
#         boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#         # Extract detection scores
#         scores = detection_graph.get_tensor_by_name('detection_scores:0')
#         # Extract detection classes
#         classes = detection_graph.get_tensor_by_name('detection_classes:0')
#         # Extract number of detectionsd
#         num_detections = detection_graph.get_tensor_by_name(
#             'num_detections:0')
#         # Actual detection.
#         (boxes, scores, classes, num_detections) = sess.run(
#             [boxes, scores, classes, num_detections],
#             feed_dict={image_tensor: image_np_expanded})
        
#         boxes = np.squeeze(boxes)
#         classes = np.squeeze(classes).astype(np.int32)
#         scores = np.squeeze(scores)

#         for i in range(len(scores)):
#             if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
#                 ymin = int(max(1,(boxes[i][0] * imH)))
#                 xmin = int(max(1,(boxes[i][1] * imW)))
#                 ymax = int(min(imH,(boxes[i][2] * imH)))
#                 xmax = int(min(imW,(boxes[i][3] * imW)))
#                 cl = int(classes[i])
#                 cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color_box[cl-1], 2)

#                 object_name = category_index[classes[i]]['name']
#                 #labels[int(classes[i])] # Look up object name from "labels" array using class index
#                 label = '%s: %d%%' % (object_name, int(scores[i]*100)) 
#                 labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
#                 label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
#                 cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
#                 cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text     
               
            

#         # Display output
#         cv2.imshow('Detect Image', frame)
        

        
        
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()