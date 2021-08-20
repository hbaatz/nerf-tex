# Config file to guide the execution of the provided code
# In general, the code is structured around modules (that themselves can contain modules) that get instantiated in a hierarchy as specified below
config = {
    # Top level module
    'module': 'network.render.Render',

    # Model & data location 
    'target_path': 'logs/carpet',
    'override': True,
    
    # Randomness seed for reproducability
    'seed': 0,

    # Dataset options
    'test_dataset_config': {                                        # Module define camera position and network inputs for test renders
        'module': 'network.dataset.Dataset',
        'data_loader_config': {
            'module': 'network.dataset.GenerateData',
            'height': 512,                                          # Height & width of the images to be rendered
            'width': 512,
            'angle': 0.55,                                          # Camera angle
            'radius': 6.,                                           # Camera radius
            'pose_dist_config': {                                   # Distribution of the camera locations scaled by the radius, pointing towards the origin
                'module': 'data.distribution.Sphere',               # In this case, uniformly spaced positions along a fixed latitude, for more options see data/distribution.py
                'sampler_config': {
                    'module': 'data.sampler.Concat',
                    'sampler_config_0': {
                        'module': 'data.sampler.Independent',
                    },
                    'sampler_config_1': {
                        'module': 'data.sampler.Grid'
                    },
                    'n': 5
                },
                'u_range': [.3,.3],
                'v_range': [0,1.]
            },
            'parameter_dist_config': {                              # Distribution of the parameters fed to the model. The position of the parameters has to match the ones from the dataset this model is trained on.
                'module': 'data.distribution.Constant',
                'constants': [[1, 1, 1, .1, 0, 0, 1]]               # In this case the parameters are [overcoat fibre length, overcoat saturation, undercoat value, ambient strength, light direciton]
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
            'b_0': [-1.5,-1.5,-1.5],
            'b_1': [1.5,1.5,1.5],
        },
        'n_epochs': 1
    },

    # Network options
    'model_config': {                                               # Module specifying the network architecture, should match the one defined for training the model
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
        'n_parameters': [1, 6]
    },

    # Rendering options
    'renderer_config': {                                            # Renderer Configuration
        'module': 'network.renderer.InstanceRenderer',
        'n_samples': 1024,                                          # Maximum number of samples along a ray
        'render_chunk': 16384,                                      # Number of rays processed in the same chunk
        'net_chunk': 32768,                                         # Number of samples fed to the network in the same chunk
        'instancer_config': {                                       # Configuration of the instancer, placing patches on a mesh
            'module': 'instancer.instancer.Instancer',
            'b_0': [-1.4,-1.2,-.1],                                 # Patch BBox size
            'b_1': [1.2,1.2,1.8],
            'cast_shadow_rays': False,                              # Wether or not to cast shadow rays
            'textures': ['meshes/smooth_checkerboard.png','','','','light'],    # Textures to specify material parameters in a spatially changing manner. They are modulated by the value provided in the test dataset config above. If no texture is provided, the value from that config is used. As a special case, lighting parameters are specified by 'light' (directional light) or 'point' (point light), and the corresponding parameters from the dataset config are interpreted as light direction or light strength and point location respectively.
            'mesh_path': 'meshes/cloth_mesh.ply',                   # Path to the mesh used for instancing
            'patch_origins_path': 'meshes/cloth_anchor_points.ply', # Path tho the anchor point locations
            'patch_scale': 0.09,                                    # Scale of the patches
            'min_shadow_samples': 8,                                # Minimal number of shadow samples per ray
            'n_shadow_samples': 256,                                # Maximal number of shadow samples per ray
            'min_texture_samples': 8,                               # Minimal number of texture samples per ray
            'n_texture_samples': 256,                               # Maximal number of texture samples per ray
            'jitter_amount': 1.,                                    # Maximal amount for jittering of patches along the normal
            'instance_sampling_method': 'nearest'                   # Method used to select patches in case of overlaps. 'nearest' selects the patch with the nearest anchor point, 'random' selects a patch at random.
        },
        'density_reweighting': True,                                # Wether to reweight the density by the invers of the number of patches intersected at a sample point
        'step_size': 0.002                                          # Step size for ray marching
    },

    'logger_config': {                                              # Logger configuration
        'module': 'network.logger.Logger'
    }
}