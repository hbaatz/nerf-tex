"""Logging utility."""

from typing import Any
import os
import time
import tensorflow as tf
from tqdm import tqdm
from util import util, interpolate

class Logger():
    """Default logger."""

    def __init__(self, target_path: str, checkpoint_variables: dict, source_path: str=None, dataset: tf.data.Dataset=None, is_training: bool=True, renderer: Any=None, n_iters: int=5e5, i_summary: int=10, i_print: int=100, i_img: int=5e3, i_checkpoint: int=1e3, max_to_keep: int=3, keep_every_n_hours: int=12, write_exr: bool=False, downsampling_factor: int=1, **kwargs) -> None:
        self.target_path = target_path
        self.source_path = source_path if source_path is not None else target_path
        self.dataset = dataset
        self.is_training = is_training
        self.renderer = renderer
        self.n_iters = n_iters
        self.i_summary = i_summary
        self.i_print = i_print
        self.i_img = i_img
        self.i_checkpoint = i_checkpoint
        self.step = checkpoint_variables.get('step', tf.Variable(0, dtype=tf.int64))
        self.time_print = time.perf_counter()
        self.write_exr = write_exr
        self.downsampling_factor = downsampling_factor

        # Initialize checkpoint loading
        checkpoint_path = os.path.join(self.source_path, 'checkpoints')
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_variables.update({'step': self.step})
        checkpoint = tf.train.Checkpoint(**checkpoint_variables)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=max_to_keep, keep_checkpoint_every_n_hours=keep_every_n_hours)

        # Load last checkpoint if available
        checkpoint.restore(self.checkpoint_manager.latest_checkpoint).expect_partial()
        if self.checkpoint_manager.latest_checkpoint:
            print('Restored model & optimizer from {}.'.format(self.checkpoint_manager.latest_checkpoint))

        if is_training:        
            # Set up summary writers
            summary_path = self.target_path # os.path.join(self.target_path, 'summaries')
            self.summary_writer = tf.summary.create_file_writer(summary_path)

            # Create path for validation images
            self.imgs_path = os.path.join(self.target_path, 'media/validation')
            os.makedirs(self.imgs_path, exist_ok=True)
        else:
            # Create path for test images
            self.imgs_path = os.path.join(self.target_path, 'media/test')
            os.makedirs(self.imgs_path, exist_ok=True)

            # Render images
            self.render_images(self.imgs_path)

    def __call__(self, loss: dict) -> None:
        self.step.assign_add(1)
        step_value = self.step.numpy()

        # Update summaries
        if step_value % self.i_summary == 0:
            with self.summary_writer.as_default():
                for key, value in loss.items():
                    tf.summary.scalar(key, value.numpy(), step = step_value)

        # Print current progress
        if step_value % self.i_print == 0:
            print('Step {}'.format(step_value), end='')
            for key, value in loss.items():
                print(' | {} {:.3g}'.format(key, value.numpy()), end='')
            print(' | Duration {:.3g}'.format(time.perf_counter() - self.time_print))
            self.time_print = time.perf_counter()

        # Render out images as specified in the val dataset
        if step_value % self.i_img == 0:
            print('Rendering validation images.')
            imgs = self.render_images(os.path.join(self.imgs_path, util.format_name('', step_value, self.n_iters, '')), True)

            with self.summary_writer.as_default():
                tf.summary.image('Validation Rendering', imgs, step_value)

        # Save checkpoints
        if step_value % self.i_checkpoint == 0:
            checkpoint_path = self.checkpoint_manager.save(checkpoint_number=step_value)
            print('Saved checkpoint to {}.'.format(checkpoint_path))

    def render_images(self, imgs_path: str, return_imgs: bool=False):
        # Create output path
        os.makedirs(imgs_path, exist_ok=True)

        # Get the dataset cardinality for name formatting purposes
        max_idx = tf.data.experimental.cardinality(self.dataset).numpy()
        if max_idx == tf.data.experimental.UNKNOWN_CARDINALITY:
            max_idx = 256

        # Iterate over the dataset and render out images
        imgs = []
        i = 0
        with tqdm(total=max_idx) as pbar: 
            for data in self.dataset:
                # Render the image for the current dataset entry
                img = self.render_image(data)

                # Get the file name
                img_name = util.format_name('', i, max_idx, '.exr' if self.write_exr else '.png')
                img_path = os.path.join(imgs_path, img_name)

                # Save the image
                self.write_image(img_path, img)

                if return_imgs:
                    imgs.append(img)

                pbar.update()
                i += 1

        if return_imgs:
            return imgs
        
    def render_image(self, data: dict):
        # Ray march along the rays specified by data
        pred = self.renderer(**data, composite_bkgd=self.dataset.composite_bkgd, bkgd_color=self.dataset.bkgd_color, training=False)

        # Reshape to 4-channel image
        img = tf.reshape(tf.concat([tf.reshape(pred['color_pred'], (-1, 3)), tf.reshape(pred['alpha_pred'], (-1, 1))], -1), (self.dataset.height, self.dataset.width, 4))

        # Filter & downsample the image
        if self.downsampling_factor > 1:
            img = interpolate.filtered_downsample(img, self.downsampling_factor)
            
        # Convert from premultiplied to non-premultiplied alpha color
        if not self.write_exr:      
            eps = 1e-5
            img = tf.concat([img[..., :3] / (img[..., 3:] + eps), img[..., 3:]], axis = -1)

        return img

    def write_image(self, img_path: str, img: tf.Tensor):
        if self.write_exr:
            import pyexr as exr
            exr.write(img_path, img.numpy())
        else:
            tf.io.write_file(img_path, tf.io.encode_png(tf.image.convert_image_dtype(img, tf.uint8)))