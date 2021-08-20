"""Collection of pixel sampling classes."""

from typing import Any
import tensorflow as tf
import numpy as np
from .ray_sampler import rays_from_camera

class Full():
    """Samples every pixel. For evaluation purposes only."""
    def __init__(self, height: int, width: int, **kwargs) -> None:
        self.height = height
        self.width = width

    def __call__(self, **kwargs) -> tf.Tensor:
        return tf.stack([tf.range(self.height * self.width) // self.width, tf.range(self.height * self.width) % self.width], -1)

class Independent():
    """Samples pixels iid."""

    def __init__(self, height: int, width: int, n_samples: int, **kwargs) -> None:
        self.height = height
        self.width = width
        self.n_samples = n_samples

    def __call__(self, **kwargs) -> tf.Tensor:
        i_samples = tf.random.uniform((self.n_samples,), maxval=self.height, dtype=tf.int32)
        j_samples = tf.random.uniform((self.n_samples,), maxval=self.width,  dtype=tf.int32)

        return tf.stack([i_samples, j_samples], -1)

class Proxy():
    """Sample pixels corresponding to rays that intersect the proxy."""

    def __init__(self, height: int, width: int, n_samples: int, proxy: Any, focal: float, downsample_factor: int=8, **kwargs) -> None:
        self.height = height
        self.width = width
        self.n_samples = n_samples
        self.proxy = proxy
        self.downsample_factor = downsample_factor
        self.focal = focal // downsample_factor
        self.height_down = height // downsample_factor
        self.width_down = width // downsample_factor

    def __call__(self, c2w: tf.Tensor) -> tf.Tensor:
        # Get rays from the camera
        image_plane_loc = tf.stack([tf.range(self.height_down * self.width_down) // self.width_down, tf.range(self.height_down * self.width_down) % self.width_down], -1)
        rays_o, rays_d, _ = rays_from_camera(tf.cast(image_plane_loc, tf.float32), self.height_down, self.width_down, self.focal, c2w)

        # Check for proxy intersection
        t = self.proxy(rays_o, rays_d)
        hit = tf.reshape(tf.where(t[:,0] == np.inf, 0, 1), (1, self.height_down, self.width_down, 1))

        if self.downsample_factor > 1:
            # Upsample
            #hit_dilate = tf.nn.max_pool(hit, ksize=3, strides=1, padding='SAME') # Create an overapproximation to not miss proxy edges, not necessary if the proxy isn't that tight anyways
            hit_dilate = hit
            hit_up = tf.image.resize(hit_dilate, (self.height, self.width), method='nearest')
        else:
            hit_up = hit

        # Get indices
        idxs = tf.where(hit_up[0,:,:,0] == 1)
        #samples = tf.random.uniform((self.n_samples,), maxval=idxs.shape[0], dtype=tf.int32) # This would be the faster way, but doesn't work with a tf.data map, since idxs.shape[0] isn't known prior to execution
        idxs = tf.random.shuffle(idxs)
        samples = tf.range(self.n_samples)

        return tf.gather_nd(idxs, samples[:, None])