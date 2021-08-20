# Config file to guide the execution of the provided code
# In general, the code is structured around modules (that themselves can contain modules) that get instantiated in a hierarchy as specified below
config = {
    # Input / output data locations
    'module': 'network.render.Render',

    # Model & data location 
    'target_path': 'logs/plush',
    'override': True,
    
    # Randomness seed for reproducability
    'seed': 0,

    # Dataset options
    'test_dataset_config': {
        'module': 'network.dataset.Dataset',
        'data_loader_config': {
            'module': 'network.dataset.GenerateData',
            'height': 800,
            'width': 800,
            'radius': 4,
            'angle': 0.63,
            'pose_dist_config': {
                'module': 'data.distribution.Sphere',
                'u_range': (.2,.2),
                'v_range': (.8,.8)
            },
            'parameter_dist_config': {
                'module': 'data.distribution.Concat',
                'distribution_config_0': {
                    'module': 'data.distribution.Constant',
                    'constants': [[1, 1]]
                }, 
                'distribution_config_1': {
                    'module': 'data.distribution.Sphere',
                    'sampler_config': {
                        'module': 'data.sampler.Concat',
                        'sampler_config_0': {
                            'module': 'data.sampler.Constant',
                            'c': .2
                        },
                        'sampler_config_1': {
                            'module': 'data.sampler.Grid',
                        },
                        'n': 5
                    },
                    'u_range': (.2,.2),
                    'v_range': (0,1)
                }
            }
        },
        'pixel_sampler_config': {
            'module': 'network.pixel_sampler.Full'
        },
        'ray_sampler_config': {
            'module': 'network.ray_sampler.Proxy',
        },
        'proxy_config': {
            'module': 'network.proxy.AABB',
            'b_0': [-.9,-.6,-.8],
            'b_1': [.9,.8,.9]
        },
        'n_epochs': 1
    },

    # Network options
    'model_config': {
        'module': 'network.model.ParamNerf',
        'pos_embedding': {
            'module': 'network.model.FourierFeatures',
            'n_freq_bands': 10
        },
        'dir_embedding': {
            'module': 'network.model.FourierFeatures',
            'n_freq_bands': 4
        },
        'param_embedding': {
            'module': 'network.model.FourierFeatures',
            'n_freq_bands': 4
        },
        'param_depth': 0,
        'color_depth': 1,
        'n_parameters': [1, 4]
    },

    # Rendering options
    'renderer_config': {
        'module': 'network.renderer.InstanceRenderer',
        'n_samples': 1280,
        'n_importance': 0,
        'perturb': False,
        'raw_noise_std': 0,
        'render_chunk': 32768,
        'net_chunk': 65536,
        'instancer_config': {
            'module': 'instancer.instancer.Instancer',
            'b_0': [-1.1,-1.1,-.2],
            'b_1': [1.1,1.1,1.1],
            'cast_shadow_rays': True,
            'textures': ['','meshes/checkerboard.png','light'],
            'mesh_path': 'meshes/stanford_bunny.ply',
            'patch_scale': 0.04,
            'min_shadow_samples': 4,
            'n_shadow_samples': 128,
            'min_texture_samples': 4,
            'n_texture_samples': 128,
            'jitter_amount': .3,
            'instance_sampling_method': 'nearest_blend'
        },
        'density_reweighting': True,
        'step_size': 0.0005
    },

    # Logging options
    'logger_config': {
        'module': 'network.logger.Logger'
    }
}