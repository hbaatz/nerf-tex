config = {
    'compute_device': 'OPTIX',                              # Device used for ray tracing, can be 'OPTIX', 'CUDA' or 'CPU'
    'seed': 0,
    'subsets': [                                            # List of dataset subsets to create
        {
            'name': 'train',                                # Subset name
            'cam_radius': 6,                                # Camera radius, used to scale position with. Camera is set to point to the origin.
            'pose_dist_config': {                           # Distribution for the camera positions
                'module': 'data.distribution.Hemisphere',   # In this case, the upper hemisphere. For more options see data/distributions.py
                'sampler_config': {
                    'module': 'data.sampler.Independent', 
                    'd': 2,
                    'n': 5000
                }
            },
            'parameter_dist_config': {                      # Distribution for the material and light parameters
                'module': 'data.distribution.Concat',       # In this case iid uniformly at random for all but the light direction, which is sampled uniformly over the surface of the unit sphere.
                'distribution_config_0': {
                    'module': 'data.distribution.AABB',
                    'sampler_config': {
                        'module': 'data.sampler.Independent',
                        'd': 4
                    }
                },
                'distribution_config_1': {
                    'module': 'data.distribution.Sphere'
                }
            }
        }
    ],
    'resolution': 512,
    'samples': 512,
    'light': 'Directional',                                 # Light source
    'collections': [                                        # Parameters to set with the distribution defined above                              
        {
            'name': 'Carpet',                               # Patch name, needs to match the name of the instancer in the corresponding .blend file
            'hair_drivers': [                               # Parameters defining the particle geometry
                'Length'
            ],
            'material_drivers': [                           # Parameters defining the particle material
                'Saturation',
                'UndercoatValue'
            ],
            'light_drivers': [                              # Parameters defining the lighting
                'Ambient',
                'LightDirection'
            ]
        }
    ],
    'pose_file_prefix': 'transforms_',                      # Prefix of the pose file
    'pose_file_save_interval': 10,                          # Number of generated samples after whicht to save poses
    'target_path': 'datasets/materials/carpet'              # Path where the generated files are stored to
}