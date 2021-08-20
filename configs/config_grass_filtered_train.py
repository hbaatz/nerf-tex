# Config file to guide the execution of the provided code
config = {
    # Top level module
    'module': 'network.train.Train',

    # Output data location 
    'target_path': 'logs/grass_filtered',
    'override': True,
    
    # Randomness seed for reproducability
    'seed': 0,

    # Dataset options
    'train_dataset_config': {
        'module': 'network.dataset.Dataset',
        'data_loader_config': {
            'module': 'network.dataset.TFRecord',
            'tfr_path': 'datasets/materials/grass_filtered/tfr/filtered.tfr'
        },
        'pixel_sampler_config': {
            'module': 'network.pixel_sampler.Proxy',
            'n_samples': 256,
        },
        'ray_sampler_config': {
            'module': 'network.ray_sampler.Proxy'
        },
        'proxy_config': {
            'module': 'network.proxy.AABB',
            'b_0': [-2.5,-2.5,-1],
            'b_1': [2.5,2.5,2.5]
        },
        'batchsize': 4,
        'shuffle_buffer_size': 100
    },
    'val_dataset_config': {
        'module': 'network.dataset.Dataset',
        'data_loader_config': {
            'module': 'network.dataset.GenerateData',
            'angle': 0.63,
            'pose_dist_config': {
                'module': 'data.distribution.Constant',
                'constants': [[.47, -.65, .6]]
            },
            'parameter_dist_config': {
                'module': 'data.distribution.Constant',
                'constants': [
                    [0,0,0,-.707,.707],                              # [filter sigma, age, light direction]
                    [10,0,0,-.707,.707]
                ]
            }
        },
        'pixel_sampler_config': {
            'module': 'network.pixel_sampler.Full'
        },
        'ray_sampler_config': {
            'module': 'network.ray_sampler.Proxy'
        },
        'proxy_config': {
            'module': 'network.proxy.AABB',
            'b_0': [-2.5,-2.5,-1],
            'b_1': [2.5,2.5,2.5]
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

    # Training options
    'loss_config': {
        'module': 'network.loss.AlphaLoss',
        'loss_fn': 'network.loss.smape',
        'alpha_loss_fn': 'network.loss.mse'
    },
    'n_iters': 500000,
    'lrate': 5e-4,
    'lrate_decay': 500,

    # Rendering options
    'renderer_config': {
        'module': 'network.renderer.Renderer',
        'n_samples': 256,
        'perturb': True,
        'raw_noise_std': 0.1,
        'render_chunk': 32768,
        'net_chunk': 65536,
        'blur_idx': 0
    },

    # Logging options
    'logger_config': {
        'module': 'network.logger.Logger'
    }
}