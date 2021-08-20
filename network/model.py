"""Collection of model architectures."""

from typing import Union
import tensorflow as tf
from tensorflow.keras.layers import Dense
from util import util, EasyDict
from network.layer import *

def Nerf(pos_embedding: EasyDict, dir_embedding: EasyDict, depth: int=8, width: int=256, skips: list=[4], name: str='model', **kwargs) -> dict:
    """The base mlp architecture as introduced by NeRF."""

    with tf.name_scope(name):
        # Initialize FFs
        pos_feature_map = util.instantiate(pos_embedding)
        dir_feature_map = util.instantiate(dir_embedding)

        # Create input layers
        pos_inputs = tf.keras.Input(shape=(3,), name='pos')
        dir_inputs = tf.keras.Input(shape=(3,), name='dir')
        # Dummy for auxilliary parameter input
        param_inputs = tf.keras.Input(shape=(None,), name='params')

        # Create feature maps of the input
        pos_inputs_map = pos_feature_map(pos_inputs)
        dir_inputs_map = dir_feature_map(dir_inputs)

        # Define MLP layers
        outputs = pos_inputs_map
        for i in range(depth):
            outputs = Dense(width, activation='relu')(outputs)
            if i in skips:
                outputs = tf.concat([pos_inputs_map, outputs], -1)

        # Density output
        alpha_outputs = Dense(1, activation=None, name='alpha')(outputs)

        # Add in directions
        outputs = Dense(width, activation=None)(outputs)
        outputs = tf.concat([dir_inputs_map, outputs], -1)

        # Color output
        outputs = Dense(width // 2, activation='relu')(outputs)
        color_outputs = Dense(3, activation=None, name='color')(outputs)

        return {name: tf.keras.Model(inputs=[pos_inputs, dir_inputs, param_inputs], outputs=[color_outputs, alpha_outputs], name=name)}

def CoarseFine(model_config: EasyDict, **kwargs) -> dict:
    """Combine two models to use one to predict sampling locations for the other as done by NeRF."""

    for key, value in kwargs.items():
        model_config.setdefault(key, value)
    model_coarse = util.instantiate(model_config)
    model_config['name'] = next(iter(model_coarse)) + '_fine'
    model_fine = util.instantiate(model_config)

    return dict(model_coarse, **model_fine)

def ParamNerf(pos_embedding: EasyDict, dir_embedding: EasyDict, param_embedding: EasyDict, n_parameters: Union[int, list], n_pos: int=3, param_depth: int=0, param_width:int = 128, depth: int=8, width: int=256, skips: list=[4], color_depth: int=1, embedding_config: EasyDict=None, include_param_dims: bool=False, name: str='model') -> dict:
    """The base mlp architecture as introduced by NeRF, with support for auxilliary parameters."""

    with tf.name_scope(name):
        # Get number of parameters to feed in at the front of the network
        if type(n_parameters) is int:
            n_parameters = [n_parameters, 0]

        # Initialize FFs
        pos_feature_map = util.instantiate(pos_embedding)
        dir_feature_map = util.instantiate(dir_embedding)
        param_feature_map = util.instantiate(param_embedding)

        # Create input layers
        pos_inputs = tf.keras.Input(shape=(n_pos,), name='pos')
        dir_inputs = tf.keras.Input(shape=(3,), name='dir')
        param_inputs = tf.keras.Input(shape=(sum(n_parameters),), name='params')

        # Create feature maps of the input
        pos_inputs_map = pos_feature_map(pos_inputs)
        dir_inputs_map = dir_feature_map(dir_inputs)

        # Initialize embedings if specified
        if embedding_config is not None:
            embedding_inputs = tf.concat([pos_inputs, param_inputs], -1) if include_param_dims else pos_inputs
            embedding = util.instantiate(embedding_config)
            embeddings = embedding(embedding_inputs)
            pos_inputs_map = tf.concat([pos_inputs_map, embeddings], -1)

        # A couple of extra MLP layers to shape the geometry parameters
        if n_parameters[0] > 0:
            param_geo_inputs_map = param_feature_map(param_inputs[:,:n_parameters[0]])
            for i in range(param_depth):
                param_geo_inputs_map = Dense(param_width, activation='relu')(param_geo_inputs_map)

            pos_inputs_map = tf.concat([pos_inputs_map, param_geo_inputs_map], -1)

        # A couple of extra MLP layers to shape the appearance parameters
        if n_parameters[1] > 0:
            param_app_inputs_map = param_feature_map(param_inputs[:,n_parameters[0]:])
            for i in range(param_depth):
                param_app_inputs_map = Dense(param_width, activation='relu')(param_app_inputs_map)

            dir_inputs_map = tf.concat([dir_inputs_map, param_app_inputs_map], -1)

        # Define MLP layers
        outputs = pos_inputs_map
        for i in range(depth):
            outputs = Dense(width, activation='relu')(outputs)
            if i in skips:
                outputs = tf.concat([pos_inputs_map, outputs], -1)

        # Density output
        alpha_outputs = Dense(1, activation=None, name='alpha')(outputs)

        # Add in directions
        outputs = Dense(width, activation=None)(outputs)
        outputs = tf.concat([dir_inputs_map, outputs], -1)

        # Extra MLP layers to map new inputs
        for i in range(color_depth):
            outputs = Dense(width, activation='relu')(outputs)

        # Color output
        outputs = Dense(width // 2, activation='relu')(outputs)
        color_outputs = Dense(3, activation=None, name='color')(outputs)

        return {name: tf.keras.Model(inputs=[pos_inputs, dir_inputs, param_inputs], outputs=[color_outputs, alpha_outputs], name=name)}