from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
"""
Created on Tue Nov 17 23:00:35 2020

@author: Angga [Modified public source]
"""

"""
# Usage:
#   # From tensorflow/models/
#   # Create train data:

####################################################################
GTSRB RGB 300x300 NoCrop
####################################################################
#python create_tfrecord.py --type=train --csv_input=Images/GTSRB/RGB_300x300_NoCrop/Train/Train.csv  --image_dir=Images/GTSRB/RGB_300x300_NoCrop/Train --output_path=data/GTSRB/train_RGB_300x300_NoCrop_GTSRB.record

#python create_tfrecord.py --type=test --csv_input=Images/GTSRB/RGB_300x300_NoCrop/Test/Test.csv  --image_dir=Images/GTSRB/RGB_300x300_NoCrop/Test --output_path=data/GTSRB/test_RGB_300x300_NoCrop_GTSRB.record
####################################################################
GTSRB RGB 300x300 Crop
####################################################################
#python create_tfrecord.py --type=train --csv_input=Images/GTSRB/RGB_300x300_Crop/Train/Train.csv  --image_dir=Images/GTSRB/RGB_300x300_Crop/Train --output_path=data/GTSRB/train_RGB_300x300_Crop_GTSRB.record

#python create_tfrecord.py --type=test --csv_input=Images/GTSRB/RGB_300x300_Crop/Test/Test.csv  --image_dir=Images/GTSRB/RGB_300x300_Crop/Test --output_path=data/GTSRB/test_RGB_300x300_Crop_GTSRB.record
####################################################################
GTSRB RGB 0x0 NoCrop
####################################################################
#python create_tfrecord.py --type=train --csv_input=Images/GTSRB/RGB_0x0_NoCrop/Train/Train.csv  --image_dir=Images/GTSRB/RGB_0x0_NoCrop/Train --output_path=data/GTSRB/train_RGB_0x0_NoCrop_GTSRB.record

#python create_tfrecord.py --type=test --csv_input=Images/GTSRB/RGB_0x0_NoCrop/Test/Test.csv  --image_dir=Images/GTSRB/RGB_0x0_NoCrop/Test --output_path=data/GTSRB/test_RGB_0x0_NoCrop_GTSRB.record
####################################################################

####################################################################
GTSDB RGB 300x300 NoCrop
####################################################################
#python create_tfrecord.py --type=train --csv_input=Images/GTSDB/RGB_300x300_NoCrop/Train/Train.csv  --image_dir=Images/GTSDB/RGB_300x300_NoCrop/Train --output_path=data/GTSDB/OneClass/train_RGB_300x300_NoCrop_GTSDB.record

#python create_tfrecord.py --type=test --csv_input=Images/GTSDB/RGB_300x300_NoCrop/Test/Test.csv  --image_dir=Images/GTSDB/RGB_300x300_NoCrop/Test --output_path=data/GTSDB/OneClass/test_RGB_300x300_NoCrop_GTSDB.record
####################################################################
GTSDB RGB 300x300 Crop
####################################################################
#python create_tfrecord.py --type=train --csv_input=Images/GTSDB/RGB_300x300_Crop/Train/Train.csv  --image_dir=Images/GTSDB/RGB_300x300_Crop/Train --output_path=data/GTSDB/OneClass/train_RGB_300x300_Crop_GTSDB.record

#python create_tfrecord.py --type=test --csv_input=Images/GTSDB/RGB_300x300_Crop/Test/Test.csv  --image_dir=Images/GTSDB/RGB_300x300_Crop/Test --output_path=data/GTSDB/OneClass/test_RGB_300x300_Crop_GTSDB.record
####################################################################
GTSDB RGB 0x0 NoCrop
####################################################################
#python create_tfrecord.py --type=train --csv_input=Images/GTSDB/RGB_0x0_NoCrop/Train/Train.csv  --image_dir=Images/GTSDB/RGB_0x0_NoCrop/Train --output_path=data/GTSDB/OneClass/train_RGB_0x0_NoCrop_GTSDB.record

#python create_tfrecord.py --type=test --csv_input=Images/GTSDB/RGB_0x0_NoCrop/Test/Test.csv  --image_dir=Images/GTSDB/RGB_0x0_NoCrop/Test --output_path=data/GTSDB/OneClass/test_RGB_0x0_NoCrop_GTSDB.record
####################################################################


# """


import os
import io
import pandas as pd
import tensorflow as tf
import sys
sys.path.append(r"C:\\RTTSR\\models\\research\\") #sesuaikan folder temen2
sys.path.append(r"C:\\RTTSR\\models\\research\\object_detection\\utils") #sesuaikan folder temen2
from PIL import Image
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
from create_pbtxt import labelText, labelOneClass

# flags = tf.compat.v1.flags
# flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
# flags.DEFINE_string('image_dir', '', 'Path to the image directory')
# flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
# FLAGS = flags.FLAGS


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        # one class
        classes_text.append(row['class_text'].encode('utf8'))
        #43 class
        
        classes.append((row['classId']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    
    return tf_example


def main(_):
    
   
     # ===========================RGB 0x0 No Crop Some============================
    # Train
    # output_path = 'data/GTSDB/SomeClass/train_RGB_0x0_NoCrop_Some.record'
    # csv_input = 'Images/GTSDB/RGB_0x0_NoCrop_Some/Train/Train.csv'
    # image_dir = 'Images/GTSDB/RGB_0x0_NoCrop_Some/Train'
    
    # Validation
    # output_path = 'data/GTSDB/SomeClass/validation_RGB_0x0_NoCrop_Some.record'
    # csv_input = 'Images/GTSDB/RGB_0x0_NoCrop_Some/Validation/Validation.csv'
    # image_dir = 'Images/GTSDB/RGB_0x0_NoCrop_Some/Validation'
    
    # Test
    # output_path = 'data/GTSDB/SomeClass/test_RGB_0x0_NoCrop_Some.record'
    # csv_input = 'Images/GTSDB/RGB_0x0_NoCrop_Some/Test/Test.csv'
    # image_dir = 'Images/GTSDB/RGB_0x0_NoCrop_Some/Test'
    # ================================================================================
    
         # ===========================RGB 300x300 No Crop Some============================
    # Train
    # output_path = 'data/GTSDB/SomeClass/train_RGB_300x300_NoCrop_Some.record'
    # csv_input = 'Images/GTSDB/RGB_300x300_NoCrop_Some/Train/Train.csv'
    # image_dir = 'Images/GTSDB/RGB_300x300_NoCrop_Some/Train'
    
    # Validation
    # output_path = 'data/GTSDB/SomeClass/validation_RGB_300x300_NoCrop_Some.record'
    # csv_input = 'Images/GTSDB/RGB_300x300_NoCrop_Some/Validation/Validation.csv'
    # image_dir = 'Images/GTSDB/RGB_300x300_NoCrop_Some/Validation'
    
    # Test
    # output_path = 'data/GTSDB/SomeClass/test_RGB_300x300_NoCrop_Some.record'
    # csv_input = 'Images/GTSDB/RGB_300x300_NoCrop_Some/Test/Test.csv'
    # image_dir = 'Images/GTSDB/RGB_300x300_NoCrop_Some/Test'
    # ================================================================================
    
    
    writer = tf.io.TFRecordWriter(output_path)
    path = os.path.join(os.getcwd(), image_dir)
    examples = pd.read_csv(csv_input,sep=';')
    # print(examples)
    grouped = split(examples, 'filename')
    # print(grouped)
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    # print(tf_example)
    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
     tf.compat.v1.app.run()
