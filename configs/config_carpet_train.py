# Config file to guide the execution of the provided code
# In general, the code is structured around modules (that themselves can contain modules) that get instantiated in a hierarchy as specified below
config = {
    # Top level module
    'module': 'network.train.Train',

    # Output data location 
    'target_path': 'logs/carpet',
    'override': True,
    
    # Randomness seed for reproducability
    'seed': 0,

    # Dataset options
    'train_dataset_config': {                                           # Module to load in training dataset images, poses and parameters
        'module': 'network.dataset.Dataset',
        'data_loader_config': {
            'module': 'network.dataset.TFRecord',
            'tfr_path': 'datasets/materials/carpet/tfr/train.tfr'       # Training dataset location
        },
        'pixel_sampler_config': {                                       # Sampler for picking locations on the image plane
            'module': 'network.pixel_sampler.Proxy',                    
            'n_samples': 256,                                           # Number of samples per image
        },
        'ray_sampler_config': {                                         # Sampler for creating rays according to the image plane locations
            'module': 'network.ray_sampler.Proxy'                       
        },
        'proxy_config': {                                               # Proxy to cull rays and define min and max locations for ray marching
            'module': 'network.proxy.AABB',
            'b_0': [-1.5,-1.3,-.2],
            'b_1': [1.3,1.3,1.9]
        },
        'batchsize': 4,                                                 # Number of images per batch
        'shuffle_buffer_size': 100
    },
    'val_dataset_config': {                                             # Module to define camera position and network inputs for validation renders
        'module': 'network.dataset.Dataset',
        'data_loader_config': {
            'module': 'network.dataset.GenerateData',
            'height': 256,                                              # Height & width of the images to be rendered
            'width': 256,
            'angle': 0.63,                                              # Camera angle
            'radius': 5.,                                               # Camera radius
            'pose_dist_config': {                                       # Distribution of the camera locations scaled by the radius, pointing towards the origin
                'module': 'data.distribution.Constant',                 # In this case, just a constant location, for more options see data/distribution.py
                'constants': [[.47, -.65, .6]]
            },
            'parameter_dist_config': {                                  # Distribution of the parameters fed to the model. The position of the parameters has to match the ones from the dataset this model is trained on.
                'module': 'data.distribution.Constant',
                'constants': [
                    [0,1,1,.1,0,-.707,.707],                            # In this case the parameters are [overcoat fibre length, overcoat saturation, undercoat brightness, ambient strength, light direciton]
                    [1,1,1,.1,0,-.707,.707]
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
            'b_0': [-1.5,-1.3,-.2],
            'b_1': [1.3,1.3,1.9]
        },
        'n_epochs': 1
    },

    # Network options
    'model_config': {                                                   # Module specifying the network architecture
        'module': 'network.model.ParamNerf',
        'pos_embedding': {                                              # Embedding configuration for the sample positions
            'module': 'network.model.FourierFeatures',
            'n_freq_bands': 10
        },
        'dir_embedding': {                                              # Embedding configuration for the sample directions
            'module': 'network.model.FourierFeatures',
            'n_freq_bands': 4
        },
        'param_embedding': {                                            # Embedding configuration for the material parameters
            'module': 'network.model.FourierFeatures',
            'n_freq_bands': 4
        },
        'n_parameters': [1, 6]                                          # Number material parameters fed with sample positions and ray directions respectively
    },

    # Training options
    'loss_config': {                                                    # Loss Configuration
        'module': 'network.loss.AlphaLoss',
        'loss_fn': 'network.loss.smape',
        'alpha_loss_fn': 'network.loss.mse'
    },
    'n_iters': 500000,
    'lrate': 5e-4,
    'lrate_decay': 500,

    # Rendering options
    'renderer_config': {                                                # Renderer Configuration
        'module': 'network.renderer.Renderer',
        'n_samples': 256,                                               # Number of samples per ray
        'perturb': True,                                                # Jitter samples on the ray
        'render_chunk': 32768,                                          # Number of rays processed in the same chunk
        'net_chunk': 65536,                                             # Number of samples fed to the network in the same chunk
    },

    # Logging options
    'logger_config': {                                                  # Logger configuration
        'module': 'network.logger.Logger'
    }
}