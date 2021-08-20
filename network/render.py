"""Render out images."""

from typing import Any
from util import util

def Render(target_path: str, 
        test_dataset_config: util.EasyDict,
        model_config: util.EasyDict,
        renderer_config: util.EasyDict,
        logger_config: util.EasyDict,
        source_path: str=None,
        override: bool=True,
        **kwargs) -> None:
    """Render images as specified in the config file."""

    # Initialize dataset
    test_dataset = util.instantiate(test_dataset_config)

    # Initialize model
    model_config.setdefault('n_parameters', test_dataset.n_parameters)
    model = util.instantiate(model_config)

    # Initialize differentiable renderer
    renderer_config.update(model)
    renderer = util.instantiate(renderer_config)

    # Load model checkpoint & render images
    logger_config.update({'target_path': target_path, 'checkpoint_variables': model, 'source_path': source_path, 'dataset': test_dataset, 'is_training': False, 'renderer': renderer})
    logger = util.instantiate(logger_config)