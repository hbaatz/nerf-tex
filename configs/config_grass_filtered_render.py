# Config file to guide the execution of the provided code
config = {
    # Input / output data locations
    'module': 'network.render.Render',

    # Model & data location 
    'target_path': 'logs/grass_filtered',
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
            'radius': {
                'module': 'data.distribution.AABB',
                'sampler_config': {
                    'module': 'data.sampler.Grid',
                    'n': 5
                },
                'b_0': 20,
                'b_1': 5
            },
            'pose_dist_config': {
                'module': 'data.distribution.Constant',
                'constants': [[0.3, -0.74, 0.6]]
            },
            'parameter_dist_config': {
                'module': 'data.distribution.Constant',
                'constants': [[.5, 0, 1, .5, .7]]
            },
            'dataset_size': 5
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
        'n_parameters': [2, 3]
    },

    # Rendering options
    'renderer_config': {
        'module': 'network.renderer.InstanceRenderer',
        'n_samples': 1024,
        'render_chunk': 16384,
        'net_chunk': 32768,
        'instancer_config': {
            'module': 'instancer.instancer.Instancer',
            'b_0': [-2,-2,-.5],
            'b_1': [2,2,2.5],
            'cast_shadow_rays': False,
            'textures': ['','','light'],
            'mesh_path': 'meshes/terrain_mesh.ply',
            'patch_origins_path': 'meshes/terrain_anchor_points.ply',
            'patch_scale': 0.1,
            'jitter_amount': 1.,
            'instance_sampling_method': 'nearest'
        },
        'step_size': 0.001,
        'blur_idx': 0
    },

    # Logging options
    'logger_config': {
        'module': 'network.logger.Logger'
    }
}