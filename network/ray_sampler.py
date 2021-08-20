"""Collection of ray sampling classes."""

from typing import Any
import tensorflow as tf

class Frustum():
    """Sample rays within the given view frustum."""

    def __init__(self, height: int, width: int, focal: float, near: float, far: float, **kwargs) -> None:
        self.height = height
        self.width = width
        self.focal = focal
        self.near = near
        self.far = far

    def __call__(self, image_plane_loc: tf.Tensor, c2w: tf.Tensor) -> tf.Tensor:
        n_samples = image_plane_loc.shape[0]
        rays_o, rays_d, cone_scale = rays_from_camera(image_plane_loc, self.height, self.width, self.focal, c2w)
        t = tf.stack((tf.repeat(self.near, repeats=n_samples), tf.repeat(self.far, repeats=n_samples)), -1)

        return rays_o, rays_d, t, cone_scale

class Proxy():
    """Sample rays within a proxy."""

    def __init__(self, height: int, width: int, focal: float, proxy: Any, **kwargs) -> None:
        self.height = height
        self.width = width
        self.focal = focal
        self.proxy = proxy

    def __call__(self, image_plane_loc: tf.Tensor, c2w: tf.Tensor) -> tf.Tensor:
        rays_o, rays_d, cone_scale = rays_from_camera(image_plane_loc, self.height, self.width, self.focal, c2w)
        rays_d = rays_d / tf.linalg.norm(rays_d, axis=-1, keepdims=True)
        t = self.proxy(rays_o, rays_d)
            
        return rays_o, rays_d, t, cone_scale

def rays_from_camera(image_plane_loc: tf.Tensor, height: int, width: int, focal: float, c2w: tf.Tensor) -> tf.Tensor:
    # Get directions from origin to image plane
    dirs = tf.stack([(image_plane_loc[:,1] + .5 - .5 * width) / focal, -(image_plane_loc[:,0] + .5 - .5 * height) / focal, -tf.ones(image_plane_loc.shape[0])], -1)
    rays_d = tf.reduce_sum(tf.expand_dims(dirs, -2) * c2w[:3, :3], -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))

    # 1. Get the un-projected radius of the cone hiting the image plane. 2. Rescale for non-uniform distance to the image plane. 3. Rescale for the actual distance on the image plane.
    cone_scale = tf.cos(tf.atan(tf.linalg.norm(dirs[:,:2], axis=-1))) / tf.linalg.norm(dirs, axis=-1) / focal

    return rays_o, rays_d, cone_scale[:, None]