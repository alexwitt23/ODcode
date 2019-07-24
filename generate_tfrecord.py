# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw COCO dataset to TFRecord for object_detection.

Please note that this tool creates sharded output files.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --image_dir="${TRAIN_IMAGE_DIR}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import math
import contextlib2
import numpy as np
import PIL.Image

import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags

tf.flags.DEFINE_string('image_dir', '',
                       'Training image directory.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

FLAGS = flags.FLAGS

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

labelsDict = {
    "0": "rock",
    "1": "motor"
}

def create_tf_example(img,
                      annotations_list,
                      image_dir,
                      category_index):
  """
  Converts image and annotations to a tf.Example proto.
  """

  with tf.io.gfile.GFile(img, 'rb') as fid:

    encoded_jpg = fid.read()

  encoded_jpg_io = io.BytesIO(encoded_jpg)

  image = PIL.Image.open(encoded_jpg_io)

  image_width, image_height = image.size

  filename = os.path.basename(img)

  xmins = []
  xmaxs = []
  ymins = []
  ymaxs = []
  category_names = []
  category_ids = []

  for object_annotations in annotations_list:

    (class_id, xmin, xmax, ymin, ymax) = tuple(object_annotations)

    xmins.append(xmin)
    xmaxs.append(xmax)
    ymins.append(ymin)
    ymaxs.append(ymax)

    category_ids.append(class_id)
    category_names.append(labelsDict[str(class_id)].encode('utf8'))

  # Create dictionary of inputs for tf.Record    
  feature_dict = {
      'image/height': dataset_util.int64_feature(image_height),
      'image/width': dataset_util.int64_feature(image_width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(category_names),
      'image/object/class/label': dataset_util.int64_list_feature(category_ids)
  }

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  
  return example


def _create_tf_record(image_dir, output_path, num_shards):

  """Loads txt info and convert to tf.Record format.

  Args:
    txt_file: txt file containing bounding box annotations.
    image_dir: Directory containing the image files.
    output_path: Path to output tf.Record file.
    num_shards: number of output file shards.
  """

  with contextlib2.ExitStack() as tf_record_close_stack:

    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(

        tf_record_close_stack, output_path, num_shards)

    imgs = num_imgs = [os.path.join(FLAGS.image_dir,img) for img in os.listdir(FLAGS.image_dir) if img.endswith('.png') or img.endswith('.jpg')]

    annotations_index = {}

    missing_annotations = 0

    for index, img in enumerate(imgs):
      # Check that for image annotation
      txt_path = img.split('.')[0] + '.txt'

      if img not in annotations_index:
        
        annotations_index[img] = []

      if os.path.exists(txt_path):

        with open(txt_path, 'r') as txt_file:

          for line in txt_file.readlines():

            shape_class, xmin, xmax, ymin, ymax = line.strip().split(' ')
            shape_class = int(shape_class)
            xmin = float(xmax)
            xmax = float(xmin)
            ymin = float(ymin)
            ymax = float(ymax)
            annotations_index[img].append((shape_class, xmin, xmax, ymin, ymax))
      else:

        missing_annotations += 1
    
    print("There are", missing_annotations, "missing annotations")

    for index, img in enumerate(imgs):

      if os.path.exists(txt_path):

        if index % 100 == 0:

          tf.compat.v1.logging.info('On image %d of %d', index, len(imgs))

        annotations_list = annotations_index[img]

        # Pass in:
          # path to img
          # annotations for this image
          # image directory
          # class dictionary

        tf_example = create_tf_example(
            img, annotations_list, image_dir, labelsDict)
        
        shard_idx = index % num_shards

        output_tfrecords[shard_idx].write(tf_example.SerializeToString())
      
    tf.compat.v1.logging.info('Finished writing')


def main(_):

  assert FLAGS.image_dir, '`image_dir` missing.'
  assert FLAGS.output_dir, '`output_dir` missing.'

  # Make output directory if not existing
  if not tf.io.gfile.isdir(FLAGS.output_dir):
    tf.io.gfile.MakeDirs(FLAGS.output_dir)
    
  train_output_path = os.path.join(FLAGS.output_dir, 'train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'val.record')

  # Get figure out how many images we have. We can used this to calculate shard num
  num_imgs = len([img for img in os.listdir(FLAGS.image_dir) if img.endswith('.png') or img.endswith('.jpg')])
  print('\nFound', num_imgs, 'images in', FLAGS.image_dir, '\n')

  num_shards = math.ceil(num_imgs / 2000)
  print('Creating', num_shards, 'shards')
  
  _create_tf_record(
      FLAGS.image_dir,
      train_output_path,
      num_shards=num_shards)
   

if __name__ == '__main__':
  tf.compat.v1.app.run()
