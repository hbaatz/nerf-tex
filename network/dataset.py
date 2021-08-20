"""Dataset loading utility."""

from typing import Tuple, Union
import os
import json
from math import tan, inf
import tensorflow as tf
from util import util, EasyDict

def Dataset(data_loader_config: EasyDict, pixel_sampler_config: EasyDict, ray_sampler_config: EasyDict=None, proxy_config: EasyDict=None, n_epochs: int=None, batchsize: int=1, shuffle_buffer_size: int=1, step: tf.Variable=None) -> tf.data.Dataset:
    """Combine image loader, pixel sampler and ray sampler into tf.data compatible dataset."""

    # Load dataset
    dataset, height, width, focal, composite_bkgd, bkgd_color = util.instantiate(data_loader_config)

    # Initialize proxy if specified
    proxy = util.instantiate(proxy_config)

    # Initialize sampling stages
    pixel_sampler_config.update({'height': height, 'width': width, 'focal': focal, 'proxy': proxy, 'step': step})
    pixel_sampler = util.instantiate(pixel_sampler_config)
    
    if ray_sampler_config is not None:
        ray_sampler_config.update({'height': height, 'width': width, 'focal': focal, 'proxy': proxy, 'step': step})
        ray_sampler = util.instantiate(ray_sampler_config)

    # Define map that converts the raw dataset (i.e. images, camera poses & parameters) to rays & reference colors
    def data_map(in_dict):
        out_dict = {'parameters': tf.cast(in_dict['parameters'], tf.float32)}

        # Sample locations on the image plane
        image_plane_loc = pixel_sampler(c2w=in_dict['pose'])

        # Get ray positions corresponding to image plane locations
        if ray_sampler_config is not None:
            rays_o, rays_d, t, cone_scale = ray_sampler(image_plane_loc=tf.cast(image_plane_loc, tf.float32), c2w=in_dict['pose'])

            out_dict.update({'rays_o': rays_o, 'rays_d': rays_d, 't': t, 'cone_scale': cone_scale})

        # Get color if dataset contains images
        if 'image' in in_dict:
            # Use image_plane_loc as index or interpolate depending on dtype
            if image_plane_loc.dtype == tf.float32:
                color = util.interpolate_img(image_plane_loc, in_dict['image'])      
            else:
                color = tf.gather_nd(in_dict['image'], image_plane_loc)
            out_dict.update({'color': color})

        if 'alpha' in in_dict:
            # Use image_plane_loc as index or interpolate depending on dtype
            if image_plane_loc.dtype == tf.float32:
                alpha = util.interpolate_img(image_plane_loc, in_dict['alpha'])      
            else:
                alpha = tf.gather_nd(in_dict['alpha'], image_plane_loc)
            out_dict.update({'alpha': alpha})

        return out_dict

    # Map dataset to its final form
    dataset_map = dataset.map(data_map, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(shuffle_buffer_size, reshuffle_each_iteration=True).repeat(n_epochs).batch(batchsize)

    # Add in some dataset specs via reflection
    # TODO: This is too hacky
    dataset_map.height = height
    dataset_map.width = width
    dataset_map.focal = focal
    dataset_map.composite_bkgd = composite_bkgd
    dataset_map.bkgd_color = bkgd_color
    # TODO: This changes the state of a generator based datset, try to avoid that
    for data in dataset_map.take(1):
        dataset_content = 'rays_o' if 'rays_o' in data else 'color'
        dataset_map.n_samples = data[dataset_content].shape[1]
        dataset_map.n_parameters = data['parameters'].shape[-1]

    return dataset_map

def TFRecord(tfr_path: str, composite_bkgd: bool=False, bkgd_color: tf.Tensor=[1,1,1.], read_exr: bool=False, compression_type: str=None) -> Tuple[tf.data.Dataset, int, int, int, bool]:
    """Create dataset from a TFRecord."""

    if os.path.isdir(tfr_path):
        paths = []
        for file_name in os.listdir(tfr_path):
            paths.append(os.path.join(tfr_path, file_name))
        tfr_path = paths

    dataset = tf.data.TFRecordDataset(tfr_path, compression_type=compression_type)

    descr = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'pose': tf.io.FixedLenFeature([], tf.string),
        'angle': tf.io.FixedLenFeature([], tf.float32),
        'parameters': tf.io.FixedLenFeature([], tf.string)
    }

    def data_map(example):
        # Parse tf.example
        features = tf.io.parse_single_example(example, descr)

        # Parse tensors from string representations
        if read_exr:
            img = tf.io.parse_tensor(features['image'], tf.float32)
            features['image'] = img[..., :3]
        else:
            img = tf.image.decode_image(features['image'], channels=4)
            img = tf.image.convert_image_dtype(img, tf.float32)
            if composite_bkgd:
                features['image'] = img[..., :3] * img[..., 3:] + (1 - img[..., 3:]) * bkgd_color
            else:
                features['image'] = img[..., :3] * img[..., 3:]
        
        features['alpha'] = img[..., 3]

        features['pose'] = tf.io.parse_tensor(features['pose'], tf.float32)

        features['parameters'] = tf.io.parse_tensor(features['parameters'], tf.float32)

        return features

    # Create dataset by mapping it using the parsing function above
    dataset_map = dataset.map(data_map)

    for data in dataset_map.take(1):
        angle = data['angle']
        height, width = data['image'].shape[:2]

    if read_exr:
        composite_bkgd = False
        
    return dataset_map, height, width, width / tan(angle / 2) / 2, composite_bkgd, bkgd_color

def FileFolder(imgs_path: str=None, poses_path: str=None, idxs: list=[], height: int=256, width: int=256, angle: float=.7, composite_bkgd: bool=False, bkgd_color: tf.Tensor=[1,1,1.]) -> Tuple[tf.data.Dataset, int, int, int, bool]:
    """Load image dataset from a folder of files according to NeRF's Blender dataset spec."""

    data = {}
    if poses_path is not None:
        poses, parameters, angle = load_poses(poses_path, idxs)
        data['pose'] = poses
        data['parameters'] = parameters
    if imgs_path is not None:
        imgs, alpha, height, width = load_imgs(imgs_path, idxs, composite_bkgd, bkgd_color)
        data['image'] = imgs
        data['alpha'] = alpha
        
    dataset = tf.data.Dataset.from_tensor_slices(data)

    return dataset, height, width, width / tan(angle / 2) / 2, composite_bkgd, bkgd_color

def load_imgs(imgs_path: str, idxs: list, composite_bkgd: bool, bkgd_color: tf.Tensor) -> Tuple[tf.Tensor, int, int]:
    """Load images and preprocess for the network."""

    img_names = [name for name in os.listdir(imgs_path) if name[-4:] in ['.png', '.jpg']]
    img_names.sort()

    imgs = []
    alpha = []
    for img_name in [name for i, name in enumerate(img_names) if i in idxs]:
        img_path = os.path.join(imgs_path, img_name)
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=4)
        img = tf.image.convert_image_dtype(img, tf.float32)

        if composite_bkgd:
            imgs.append(img[..., :3] * img[..., 3:] + (1 - img[..., 3:])) * bkgd_color
        else:
            imgs.append(img[..., :3] * img[..., 3:])
            
        alpha.append(img[..., 3])

    imgs = tf.stack(imgs)
    alpha = tf.stack(alpha)

    return imgs, alpha, imgs.shape[1], imgs.shape[2]

def load_poses(pose_path: str, idxs: list) -> Tuple[tf.Tensor, tf.Tensor, float]:
    """Load poses."""

    with open(pose_path) as pose_file:
        pose_dict = json.load(pose_file)

    poses = []
    parameters = []
    # NOTE: This assumes dict to be order preserving (which holds for python>=3.7)
    for pose in [p for i, p in enumerate(pose_dict['frames']) if i in idxs]:
        poses.append(pose['transform_matrix'])
        if 'driver_parameters' in pose:
            # sorted_parameters = []
            # for driver in sorted(pose['driver_parameters']):
            #    sorted_parameters.append(pose['driver_parameters'][driver])
            # parameters.append(sorted_parameters)
            
            # Don't sort, take order from pose file
            parameters.append(list(pose['driver_parameters'].values()))
        else:
            parameters.append([])

    return tf.constant(poses), tf.constant(parameters), pose_dict['camera_angle_x']

def GenerateData(height: int=256, width: int=256, angle: float=.7, pose_dist_config: EasyDict=EasyDict({'module': 'data.dist.Hemisphere'}), radius: Union[float, EasyDict]=5., offset: list=[0.,0.,0.], parameter_dist_config: EasyDict=EasyDict({'module': 'data.distribution.Constant'}), dataset_size: int=-1, composite_bkgd: bool=False, bkgd_color: tf.Tensor=[1, 1, 1.]) -> Tuple[tf.data.Dataset, int, int, int, bool]:
    """Sample camera poses as set by pose_dist_config looking towards the origin and parameters from the distribution specified by parameter_dist."""

    pose_dist = util.instantiate(pose_dist_config)

    param_dist = util.instantiate(parameter_dist_config)

    if isinstance(radius, dict):
        rad = util.instantiate(radius)
    else:
        rad = lambda: radius
        

    # Get minimal dataset size out of all three specs
    min_dataset_size = max([dataset_size, pose_dist.sampler.n, param_dist.sampler.n])
    
    # If the min_dataset_size < 256 create as precomputed data
    if (min_dataset_size <= 256):
        data = {'pose': [], 'parameters': []}
        for _ in range(min_dataset_size):
            data['pose'].append(look_at(pose_dist() * rad(), offset=tf.constant(offset)))
            data['parameters'].append(param_dist())
        
        dataset = tf.data.Dataset.from_tensor_slices(data)
    else:
        def generator():
            while True:
                yield {'pose': look_at(pose_dist() * rad()), 'parameters': param_dist()}

        dataset = tf.data.Dataset.from_generator(generator, output_types={'pose': tf.float32, 'parameters': tf.float32}).take(min_dataset_size)

    return dataset, height, width, width / tan(angle / 2) / 2, composite_bkgd, bkgd_color

def look_at(pos, to=tf.constant([0.,0.,0.]), offset=tf.constant([0.,0.,0.]), eps=1e-6) -> tf.Tensor:
    """Create a look-at camera transform."""

    v_forward, _ = tf.linalg.normalize(pos - to + eps)
    v_right, _ = tf.linalg.normalize(tf.linalg.cross([0,0,1.], v_forward) + eps)
    v_up, _ = tf.linalg.normalize(tf.linalg.cross(v_forward, v_right) + eps)

    return tf.concat([tf.stack([v_right, v_up, v_forward, pos + offset], axis=1), [[0,0,0,1.]]], axis=0)