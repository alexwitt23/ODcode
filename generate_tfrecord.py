'''
Author: Alex Witt

Written for TerraClear

Read the txt files and images in from their directories and convert to tf.Records
'''

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import glob
import math
import multiprocessing
from multiprocessing import Lock
import numpy as np
from tqdm import tqdm

import contextlib2
import tensorflow as tf


from PIL import Image
from object_detection.utils import dataset_util

from object_detection.dataset_tools import tf_record_creation_util

from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('img_dir', '', 'Path to images')
FLAGS = flags.FLAGS


def create_tf_examples(path_to_imgs):

    # Get images
    img_paths = [os.path.join(path_to_imgs,file) for file in os.listdir(path_to_imgs) if file.endswith(".jpg") or file.endswith(".png")]
    # Get the txts and also check that a txt exists for the img 
    full_img_paths = []
    full_txt_paths = []
    for path in img_paths:
        txt = path.split('.')[0]
        txt = txt + ".txt"
      
        if os.path.exists(os.path.join(path_to_imgs,txt)):
            full_img_paths.append(os.path.join(path_to_imgs,path))
            full_txt_paths.append(os.path.join(path_to_imgs,txt))
        else:
            continue

    sorted(full_img_paths, key=lambda i: str(os.path.splitext(os.path.basename(i))[0]))
    sorted(full_txt_paths, key=lambda i: str(os.path.splitext(os.path.basename(i))[0]))

    num_shards = math.ceil(len(img_paths)/2000)  

    shard_nums = np.random.choice(num_shards, len(img_paths), replace=True)

    output_filebase=os.path.join(os.getcwd(),"training/train_dataset.record")
    
    # make training data dir
    os.makedirs(output_filebase, exist_ok=True)

    data = zip(full_img_paths,full_txt_paths)

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filebase, num_shards)
        index = 0
        for index, data in enumerate(data):
            tf_example = _create_tf_example(data)
            shard_num = index % num_shards
            output_tfrecords[shard_num].write(tf_example.SerializeToString())
            index += 1

def _create_tf_example (data):
    img_path, txt_path =  data

    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = img_path.encode('utf8')
    image_format = b'png'

    image_data = []
    with open(txt_path, 'r') as dims:
        for line in dims.readlines():
            class_num, x, y, w, h = line.strip().split(' ')
            x, y, w, h = float(x), float(y), float(w), float(h)
            image_data.append((class_num, x, y, w, h))

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for data in image_data:
        xmins.append(data[1])
        xmaxs.append(data[1]+data[3])
        ymins.append(data[2])
        ymaxs.append(data[2]+data[4])
        classes_text.append("rock".encode('utf8'))
        classes.append(0)

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

    path_to_imgs = os.path.join(FLAGS.img_dir)
    create_tf_examples(path_to_imgs)

if __name__ == '__main__':
    tf.app.run()
