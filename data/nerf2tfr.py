from typing import Tuple
import os
import argparse
import math
import tensorflow as tf
import json
from tqdm import tqdm

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def load_poses(pose_path: str, skip_params: bool) -> Tuple[tf.Tensor, tf.Tensor, float]:
    """Loads poses from file."""

    with open(pose_path) as pose_file:
        pose_dict = json.load(pose_file)

    poses = []
    parameters = []
    for pose in pose_dict['frames']:
        poses.append(pose['transform_matrix'])
        if 'driver_parameters' in pose and not skip_params:
            # sorted_parameters = []
            # for driver in sorted(pose['driver_parameters']):
            #     sorted_parameters.append(pose['driver_parameters'][driver])
            # parameters.append(sorted_parameters)

            # Don't sort, take order from pose file
            parameters.append(list(pose['driver_parameters'].values()))
        else:
            parameters.append([])

    return tf.constant(poses), tf.constant(parameters), pose_dict['camera_angle_x']

def compile_feature(img_path, pose, angle, parameters) -> tf.train.Example:
    """Compile inputs into tf.train.Example feature."""

    _, img_extension = os.path.splitext(img_path)
    if img_extension == '.png':
        img = tf.io.read_file(img_path)
    elif img_extension == '.exr':
        import pyexr as exr
        img = tf.io.serialize_tensor(exr.read(img_path))
    else:
        raise ValueError('Unknown filetype.')

    feature = {
        'image': _bytes_feature(img),
        'pose': _bytes_feature(tf.io.serialize_tensor(pose)),
        'angle': _float_feature(angle),
        'parameters': _bytes_feature(tf.io.serialize_tensor(parameters))
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts NeRF dataset to TFR dataset.')
    parser.add_argument('path_in', help='Path to NeRF dataset.')
    parser.add_argument('path_out', help='Path to save TFR dataset to.')
    parser.add_argument('--subsets', nargs='+', default=['train'], help='Subsets to process.')
    parser.add_argument('--skip_params', action='store_true', help='Do not include hair parameters in the tfr file.')
    parser.add_argument('--imgs_per_shard', type=int, default=-1, help='Number of images per shard.')
    parser.add_argument('--compression_type', type=str, default='', help='Compression used for the tfrecords.')
    args = parser.parse_args()

    # Create output dir
    os.makedirs(args.path_out)
 
    # Copy images and poses from test and val to single target
    offset = 0
    for subset in args.subsets:
        print('Processing {} subset.'.format(subset))

        # Get image filenames
        imgs_path = os.path.join(args.path_in, subset)
        img_names = sorted(os.listdir(imgs_path))
        n_imgs = len(img_names)

        # Get pose file path
        path_transforms = os.path.join(args.path_in, 'transforms_' + subset + '.json')
        poses, parameters, angle = load_poses(path_transforms, args.skip_params)

        # Get number of shards
        if args.imgs_per_shard < 0:
            args.imgs_per_shard = n_imgs 
        n_shards = math.ceil(n_imgs / args.imgs_per_shard)

        options = tf.io.TFRecordOptions(compression_type=args.compression_type)
        with tqdm(total = len(img_names)) as pbar:
            for shard in range(n_shards):
                # Get output file path
                if n_shards == 1:
                    suffix = ''
                else:
                    suffix = '_' + str(shard)

                path_out_subset = os.path.join(args.path_out, subset + suffix + '.tfr')

                # Write to shard
                with tf.io.TFRecordWriter(path_out_subset, options=options) as writer:
                    for i in range(shard * args.imgs_per_shard, min((shard + 1) * args.imgs_per_shard, n_imgs)):
                        img_path = os.path.join(imgs_path, img_names[i])
                        example = compile_feature(img_path, poses[i], angle, parameters[i])
                        writer.write(example.SerializeToString())
                            
                        pbar.update(1)