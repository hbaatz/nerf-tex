import sys
sys.path.append('.')

import os
import argparse
import math
import numpy as np
from skimage import io, filters, util
import json
from tqdm import tqdm
from util import interpolate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Applies blur with random sigma to images and saves the amount to pose file.')
    parser.add_argument('path_in', help='Path to NeRF dataset.')
    parser.add_argument('path_out', help='Path to save to.')
    parser.add_argument('--subsets', nargs='+', default=['train'], help='Subsets to process.')
    parser.add_argument('--max_sigma', type=float, default=0, help='Max sigma value.')
    parser.add_argument('--dataset_size_increase', type=int, default=1, help='Integer factor by which to increase the dataset.')
    parser.add_argument('--p', type=float, default=3, help='Exponent stearing the sigma distribution.')
    args = parser.parse_args()

    # Create output dir
    os.makedirs(args.path_out)

    # Loop over images
    for subset in args.subsets:
        print('Processing {} subset.'.format(subset))

        # Get image filenames
        imgs_path = os.path.join(args.path_in, subset)
        img_names = sorted(os.listdir(imgs_path))
        n_imgs = len(img_names)
        n_imgs_out = n_imgs * args.dataset_size_increase

        # Get pose file path
        path_pose = os.path.join(args.path_in, 'transforms_' + subset + '.json')
        with open(path_pose) as pose_file:
            pose_dict = json.load(pose_file)

        # Get image output file path
        path_out_subset = os.path.join(args.path_out, subset)
        os.makedirs(path_out_subset)

        # Inverse cdf of exponential distribution for sampling
        def inv_cdf(x, p):
            if -1e-4 < p and p < 1e-4:
                return x
            else:
                return -np.log(1 - x * (1 - np.exp(-p))) / p

        # Sample random sigma values
        np.random.seed(0)
        p = 3
        samples = inv_cdf(np.random.rand(n_imgs_out), p)
        sigma = (samples * args.max_sigma).tolist()

        # Get format string for image names
        min_chars = math.ceil(np.log10(n_imgs_out))
        format_str = '{:0' + str(min_chars) + 'd}'

        # Define map
        def blur_img(inp):
            idx, img_name, sigma = inp
            format = os.path.splitext(img_name)[-1]

            if format == '.png':
                # Load image
                img = io.imread(os.path.join(imgs_path, img_name))
                img = util.img_as_float(img)
                img[:,:,:3] = (img[:,:,:3] ** 2.2 * (img[:,:,3:]))
                img = filters.gaussian(img, sigma=sigma, mode='constant', multichannel=True)
                img[:,:,:3] = (img[:,:,:3] / (img[:,:,3:] + 1e-5)) ** (1 / 2.2)

                # Save image
                img_prefix = img_name.split('_')[0]
                img_name_out = img_prefix + '_' + format_str.format(idx) + '.png'

                io.imsave(os.path.join(path_out_subset, img_name_out), util.img_as_ubyte(np.clip(img,0,1)), check_contrast=False)
            elif format == '.exr':
                import pyexr as exr

                # Load image
                img = exr.read(os.path.join(imgs_path, img_name))

                # Apply blur
                img = interpolate.filtered_downsample(img, 1, sigma).numpy()

                # Save image
                img_prefix = img_name.split('_')[0]
                img_name_out = img_prefix + '_' + format_str.format(idx) + '.exr'

                exr.write(os.path.join(path_out_subset, img_name_out), img)
            else:
                raise ValueError('Unknown filetype.')

        with tqdm(total=n_imgs_out) as pbar:
            for inp in zip(range(n_imgs_out), img_names * args.dataset_size_increase, sigma):
                blur_img(inp)
                pbar.update()

        # Get output pose dict
        pose_dict_out = {}
        pose_dict_out.update({'camera_angle_x': pose_dict['camera_angle_x'], 'frames': []})

        # Add sigma values to pose dict
        for i in range(n_imgs_out):
            pose_dict_out['frames'].append(dict(pose_dict['frames'][i % n_imgs]))

            img_path = pose_dict_out['frames'][i]['file_path'].split('_')[0]
            img_path_out = img_path + '_' + format_str.format(i)
            pose_dict_out['frames'][i]['file_path'] = img_path_out

            updated_parameters = {'Blur': sigma[i]}
            updated_parameters.update(pose_dict_out['frames'][i]['driver_parameters'])
            pose_dict_out['frames'][i]['driver_parameters'] = updated_parameters

        # Save pose dict to new location
        path_out_pose = os.path.join(args.path_out, 'transforms_' + subset + '.json')
        with open(path_out_pose, 'w+') as pose_file:
            json.dump(pose_dict_out, pose_file, sort_keys=False, indent=4)