"""Training loop."""

import os
import tensorflow as tf
from util import util, EasyDict

def Train(target_path: str, 
        train_dataset_config: EasyDict,
        val_dataset_config: EasyDict,
        model_config: EasyDict,
        loss_config: EasyDict, 
        n_iters: int,
        lrate: float,
        lrate_decay: float,
        renderer_config: EasyDict,
        logger_config: EasyDict,
        **kwargs) -> None:
    """Setup and run supervised training."""

    # Get variable to track the global step
    step = tf.Variable(0, dtype=tf.int64)

    # Initialize datasets
    train_dataset_config.update({'step': step})
    train_dataset = util.instantiate(train_dataset_config)

    val_dataset = util.instantiate(val_dataset_config)

    # Initialize model, get parameters and print layer structure
    model_config.setdefault('n_parameters', train_dataset.n_parameters)
    model = util.instantiate(model_config)

    model_params = []
    for m in model.values(): model_params += m.trainable_variables

    for key, m in model.items(): 
        m.summary()
        plot_path = os.path.join(target_path, key + '.png')
        tf.keras.utils.plot_model(m, plot_path, show_shapes=True)

    # Initialize differentiable renderer
    renderer_config.update(model)
    renderer = util.instantiate(renderer_config)

    # Intialize loss function
    loss_fn = util.instantiate(loss_config)

    # Set up optimizer
    if lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate, decay_steps=lrate_decay * 1e3, decay_rate=0.1)

    optimizer = tf.keras.optimizers.Adam(lrate)

    # Set up logger, load checkpoint if available
    checkpoint_variables = dict(model, step=step, optimizer=optimizer)
    logger_config.update({'target_path': target_path, 'checkpoint_variables': checkpoint_variables, 'dataset': val_dataset, 'renderer': renderer, 'n_iters': n_iters})
    logger = util.instantiate(logger_config)

    # Training loop
    for data in train_dataset.take(n_iters - logger.step):
        # Training Step
        with tf.GradientTape() as tape:
            pred = renderer(**data, composite_bkgd=train_dataset.composite_bkgd, bkgd_color=train_dataset.bkgd_color)
            loss = loss_fn(color_true=data['color'], alpha_true=data['alpha'], **pred)
            
        gradients = tape.gradient(loss, model_params)
        optimizer.apply_gradients(zip(gradients, model_params))

        # Logging
        logger({'Loss': loss})