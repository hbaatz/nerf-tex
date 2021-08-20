import bpy
from mathutils import Vector
import os
import sys
import argparse
import importlib
import hashlib
import math
import numpy as np
import json
import copy

# This is neccessary as this script is run within Blender's own environment
blend_dir = os.path.dirname(bpy.data.filepath)
if blend_dir not in sys.path:
   sys.path.append(os.path.dirname(blend_dir))

from util import util, EasyDict

def set_seed(identifier):
    """Get device indpenedent seed."""
    config_hash = hashlib.sha1(identifier.encode('UTF-8')).hexdigest()
    np.random.seed(int(config_hash[:7], 16))

def get_cam_name(i, min_chars=7):
    """Create formatted camera name from index."""
    format_str = '{:0' + str(min_chars) + 'd}'
    return 'cam_' + format_str.format(i)

def matrix2list(mat):
    """Create list of lists from blender Matrix type."""
    return list(map(list, list(mat)))

def render_views():
    # Get the location of this script to create files relative to it later on
    path_script = os.path.dirname(__file__)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Create dataset from materials.blend as specified in the config file.')
    parser.add_argument('config', help='Path to config file.')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])

    # Clip away .py ending if neccessary and replace / by .
    config_path = args.config[:-3] if args.config[-3:] == '.py' else args.config
    config_module = config_path.replace('/', '.')

    # Import config file
    config = EasyDict(importlib.import_module(config_module).config)

    # Create a folder for the dataset and save a copy of the configs
    dataset_dir = config.target_path #os.path.join(path_script, config.target_path)
    os.makedirs(dataset_dir, exist_ok=True)

    path_config_save = os.path.join(dataset_dir, 'config.json')
    with open(path_config_save, 'w+') as config_file:
        json.dump(config, config_file, indent=4)

    # Configure Blender
    if 'resolution' in config:
        bpy.context.scene.render.resolution_x = config.resolution
        bpy.context.scene.render.resolution_y = config.resolution

    if 'samples' in config:
        bpy.context.scene.cycles.samples = config['samples']
        bpy.context.scene.eevee.taa_samples = config['samples'] # TODO: That is not set for some reason...

    if 'light' in config:
        for light in bpy.context.scene.view_layers[0].layer_collection.children['Scene Stuff'].children['Light'].children:
            if light.name == config.light:
                light.exclude = False
            else:
                light.exclude = True

    image_settings = bpy.context.scene.render.image_settings
    image_settings.file_format = 'PNG'
    file_ending = '.png'
    if 'file_format' in config:
        if config['file_format'] == 'exr':
            image_settings.file_format = 'OPEN_EXR'
            image_settings.color_depth = '32'
            file_ending = '.exr'

    if 'ambient_light_strength' in config:
        bpy.data.worlds['World'].node_tree.nodes['Background'].inputs['Strength'].default_value = config.ambient_light_strength

    # Make sure to use the all GPUs available
    log_path = os.path.join(dataset_dir, 'info.log')
    with open(log_path, 'w+') as log_file:
        blender_preferences = bpy.context.preferences.addons['cycles'].preferences
        blender_preferences.compute_device_type = config['compute_device']
        for devices in blender_preferences.get_devices():
            for device in devices:
                if device.type == 'CPU':
                    device.use == False
                else:
                    device.use == True
                print("Device '{}' - {}: {}.".format(device.name, device.type, 'enabled' if device.use else 'disabled'))
                log_file.write("Device '{}' - {}: {}.\n".format(device.name, device.type, 'enabled' if device.use else 'disabled'))
        bpy.context.scene.cycles.device = 'GPU'

    # Get reference camera and add 'Cameras' collection
    cam_reference_name = config['cam_name'] if 'cam_name' in config else 'Camera'
    cam_reference = bpy.data.cameras[cam_reference_name]
    cam_collection = bpy.data.collections.new(name='Cameras')
    bpy.context.scene.collection.children.link(bpy.data.collections['Cameras'])

    # Create camera to render with from reference
    # Create and add to 'Cameras' collection
    cam_name = 'cam'
    cam = bpy.data.cameras.new(cam_name)
    cam_object = bpy.data.objects.new(cam_name, cam)
    cam_collection.objects.link(cam_object)

    # Copy properties of 'Camera' object in materials.blend
    cam.angle = config['angle'] if 'angle' in config else cam_reference.angle

    # Set as camera to render with
    bpy.context.scene.camera = cam_object

    # Create separate sets for training, validation and testing
    for subset in config.subsets:
        # Init distribution
        distribution = util.instantiate(subset['pose_dist_config'])

        # Init driver sampler
        driver_sampler = util.instantiate(subset['parameter_dist_config'])

        # Init offset to split dataset creation among multiple machines
        offset = 0
        if 'offset' in config:
            offset = config['offset']

        # Check if pose file already exists, if so append, else create a new one
        path_transforms = os.path.join(dataset_dir, config.pose_file_prefix + subset['name'] + '.json')

        if os.path.exists(path_transforms):
            with open(path_transforms) as pose_file:
                transforms = json.load(pose_file)
            offset += len(transforms['frames'])
            distribution.sampler.idx = offset
            driver_sampler.idx = offset
        else:
            transforms = {}
            transforms['camera_angle_x'] = cam_reference.angle_x
            transforms['frames'] = []
            offset += 0

        # Get collection containing the objects to render
        object_collection = bpy.context.scene.collection.children[0]
        view_layer_ref = bpy.context.scene.view_layers[0].layer_collection.children['Materials']

        # Make sure all objects are per default invisible
        for obj in view_layer_ref.children:
            obj.exclude = True

        # Create views
        i = 0
        n_samples = max(distribution.sampler.n, driver_sampler.sampler.n)
        while not (distribution.sampler.done() or driver_sampler.sampler.done()):
            print('### RENDERING IMAGE {} / {} ###\n'.format(i + offset, n_samples))

            # Init the random sampler, platform independently
            set_seed(str(config.seed) + subset['name'] + str(i + offset))

            cam_name = get_cam_name(i + offset, math.ceil(np.log10(n_samples)))

            # Translate to a point given by the specified distribution on a sphere of radius 'cam_radius' and point to origin
            cam_object.location = subset['cam_radius'] * distribution()
            cam_direction = -cam_object.location
            cam_rot_quat = cam_direction.to_track_quat('-Z', 'Y')
            cam_object.rotation_euler = cam_rot_quat.to_euler()
            if 'cam_offset' in subset:
                cam_object.location += Vector(subset['cam_offset'])
            bpy.context.view_layer.update()

            # Sample collection to render
            collection_args = config.collections[np.random.choice(len(config.collections))]
            obj_name = collection_args['name']
            obj = view_layer_ref.children[obj_name]

            # Sample driver setting
            param_sample = driver_sampler()
            print('DRIVERS: {}'.format(param_sample))

            # Dict to collect driver samples
            driver_params = {}

            # Set drivers according to sample
            idx = 0

            # Set specified hair drivers uniormly at random
            for driver in collection_args['hair_drivers']:
                bpy.data.particles[obj_name][driver] = param_sample[idx]
                driver_params[driver] = param_sample[idx]
                print('SET {} to {}'.format(driver, param_sample[idx]))
                idx += 1

            # Material drivers
            for driver in collection_args['material_drivers']:
                bpy.data.objects[obj_name].material_slots[0].material[driver] = param_sample[idx]
                driver_params[driver] = param_sample[idx]
                print('SET {} to {}'.format(driver, param_sample[idx]))
                idx += 1
            
            # Light drivers
            for driver in collection_args['light_drivers']:
                if driver in ['LightDirection', 'lightPosition']:
                    bpy.data.objects[config['light']]['x'] = param_sample[idx]
                    bpy.data.objects[config['light']]['y'] = param_sample[idx+1]
                    bpy.data.objects[config['light']]['z'] = param_sample[idx+2]
                    
                    driver_params['LightX'] = param_sample[idx]
                    driver_params['LightY'] = param_sample[idx+1]
                    driver_params['LightZ'] = param_sample[idx+2]
                    idx += 3
                else:
                    bpy.data.lights[config['light']][driver] = param_sample[idx]
                    driver_params[driver] = param_sample[idx]
                    # print('SET {} to {}'.format(driver, param_sample[idx]))
                    idx += 1

            # Create folder for the object
            path_dir = os.path.join(dataset_dir, subset['name'])
            os.makedirs(path_dir, exist_ok=True)

            # Render object and save to file
            obj.exclude = False
            bpy.context.scene.render.filepath = os.path.join(path_dir, cam_name + file_ending)
            bpy.ops.render.render(write_still=True)
            obj.exclude = True

            # Add pose to dict
            transforms['frames'].append({
                'file_path': './' + subset['name'] + '/' + cam_name,
                #'rotation': 0, # Commented out as it isn't actually used in NeRF's implementation
                'transform_matrix': matrix2list(cam_object.matrix_world),
                'driver_parameters': driver_params
            })

            # Save pose dict to file at specified intervals
            if 'pose_file_save_interval' in config and (i + 1) % config['pose_file_save_interval'] == 0:
                with open(path_transforms, 'w+') as pose_file:
                    json.dump(transforms, pose_file, sort_keys=False, indent=4)

            i += 1

        # Write the remaining poses to file
        with open(path_transforms, 'w+') as pose_file:
            json.dump(transforms, pose_file, sort_keys=False, indent=4)

if __name__ == '__main__':
    render_views()