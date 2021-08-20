"""Collection of simple proxy types with intersection methods."""

import tensorflow as tf
import numpy as np

class AABB():
    """A axis aligned bounding box, defined in [x_0,x_1]x[y_0,y_1]x[z_0,z_1] for b_0=[x_0,y_0,z_0], b1=[x_1,y_1,z_1]."""
    
    def __init__(self, b_0: list, b_1: list):
        self.b_0 = tf.constant(b_0, dtype=tf.float32)
        self.b_1 = tf.constant(b_1, dtype=tf.float32)

    def __call__(self, rays_o: tf.Tensor, rays_d: tf.Tensor) -> tf.Tensor:
        """Intersect rays with proxy, assumes that the ray origin is outside of the AABB."""

        # Get ray plane intersection
        inv_rays_d = 1. / rays_d
        t_0 = (self.b_0 - rays_o) * inv_rays_d
        t_1 = (self.b_1 - rays_o) * inv_rays_d

        # Reorder such that the nearer intersection is in t_0
        t_0_tmp = t_0
        t_0 = tf.where(t_0 < t_1, t_0, t_1)
        t_1 = tf.where(t_0_tmp > t_1, t_0_tmp, t_1)

        # Get box intersection from plane intersections 
        t_0 = tf.reduce_max(t_0, axis=1)
        t_1 = tf.reduce_min(t_1, axis=1)
        
        # Set both t_0, t_1 to infinity if there is no intersection
        t_0_tmp = t_0
        t_0 = tf.where(t_0 < t_1, t_0, np.inf)
        t_1 = tf.where(t_0_tmp < t_1, t_1, np.inf)

        return tf.stack([t_0, t_1], -1)