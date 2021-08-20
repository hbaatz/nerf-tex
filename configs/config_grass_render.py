# Config file to guide the execution of the provided code
config = {
    # Top level module
    'module': 'network.render.Render',

    # Model & data location 
    'target_path': 'logs/grass',
    'override': True,
    
    # Randomness seed for reproducability
    'seed': 0,

    # Dataset options
    'test_dataset_config': {
        'module': 'network.dataset.Dataset',
        'data_loader_config': {
            'module': 'network.dataset.GenerateData',
            'height': 512,
            'width': 512,
            'angle': 0.5,
            'radius': 6.,
            'pose_dist_config': {
                'module': 'data.distribution.Constant',
                'constants': [[0.30614675, -0.73910363, 0.6]]
            },
            'parameter_dist_config': {
                'module': 'data.distribution.Concat',
                'distribution_config_0': {
                    'module': 'data.distribution.Constant',
                    'constants': [[0, 0.33]]
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
                    'u_range': [.2,.2],
                    'v_range': [0, 1.]
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
            'b_0': [-1.2,-1.2,-.1],
            'b_1': [1.2,1.2,1]
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
        'n_parameters': [1, 4]
    },

    # Rendering options
    'renderer_config': {
        'module': 'network.renderer.InstanceRenderer',
        'n_samples': 1024,
        'render_chunk': 16384,
        'net_chunk': 32768,
        'instancer_config': {
            'module': 'instancer.instancer.Instancer',
            'b_0': [-1.6,-1.6,-.1],
            'b_1': [1.8,1.9,1.3],
            'cast_shadow_rays': True,
            'textures': ['','point'],
            'mesh_path': 'meshes/terrain_mesh.ply',
            'patch_origins_path': 'meshes/terrain_anchor_points.ply',
            'patch_scale': 0.1,
            'min_shadow_samples': 8,
            'n_shadow_samples': 128,
            'jitter_amount': 1.,
            'instance_sampling_method': 'nearest'
        },
        'step_size': 0.001
    },

    # Logging options
    'logger_config': {
        'module': 'network.logger.Logger'
    }
}