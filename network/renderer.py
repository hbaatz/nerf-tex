"""Differentiable ray marcher."""

# MIT License

# Copyright (c) 2020 bmild

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Any, Tuple
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from util import util, EasyDict

class Renderer():
    """Differentiable ray marcher, derived from the original implementation by Ben Mildenhall et al., https://github.com/bmild/nerf."""

    def __init__(self, model: tf.keras.Model, model_fine: tf.keras.Model=None, n_samples: int=64, n_importance: int=0, perturb: bool=True, raw_noise_std: float=0, render_chunk: int=32768, net_chunk: int=65536, downsampling_factor: int=1, blur_idx: int=None, map_exr: bool=False, **kwargs) -> None:
        self.model = model
        self.model_fine = model_fine
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.perturb = perturb
        self.raw_noise_std = raw_noise_std
        self.render_chunk = render_chunk
        self.net_chunk = net_chunk
        self.downsampling_factor = downsampling_factor
        self.blur_idx = blur_idx
        self.map_exr = map_exr

    def __call__(self, rays_o: tf.Tensor, rays_d: tf.Tensor, t: tf.Tensor, parameters: tf.Tensor, cone_scale: tf.Tensor, composite_bkgd: bool=False, bkgd_color: tf.Tensor=[1, 1, 1.], training: bool=True, **kwargs) -> dict:
        """Split batch into render chunks, render each and combine in the end."""

        # Flatten input along batch dimension
        rays_o_flat = tf.reshape(rays_o, [-1] + list(rays_o.shape[2:]))
        rays_d_flat = tf.reshape(rays_d, [-1] + list(rays_d.shape[2:]))
        t_flat = tf.reshape(t, [-1] + list(t.shape[2:]))
        parameters_flat = tf.repeat(parameters, repeats=rays_o.shape[1], axis=0)
        cone_scale_flat = tf.reshape(cone_scale,  [-1] + list(cone_scale.shape[2:]))

        # Filter out rays not hitting the scene bounds (i.e. rays for which t = inf)
        idxs = tf.where(t_flat[:,0] != np.inf)
        idxs_inv = tf.where(t_flat[:,0] == np.inf)
        rays_o_gather = tf.gather_nd(rays_o_flat, idxs)
        rays_d_gather = tf.gather_nd(rays_d_flat, idxs)
        t_gather = tf.gather_nd(t_flat, idxs)
        if parameters_flat.shape[1] > 0:
            parameters_gather = tf.gather_nd(parameters_flat, idxs)
        else:
            parameters_gather = tf.zeros((idxs.shape[0], 0), dtype=tf.float32)
        cone_scale_gather = tf.gather_nd(cone_scale_flat, idxs)

        # Render chunks
        out = {}
        #for i in tqdm(tf.range(0, rays_o_gather.shape[0], self.render_chunk), leave=False):
        for i in tf.range(0, rays_o_gather.shape[0], self.render_chunk):
            out_chunk = self.render_rays(rays_o=rays_o_gather[i:i+self.render_chunk], rays_d=rays_d_gather[i:i+self.render_chunk], t=t_gather[i:i+self.render_chunk], parameters=parameters_gather[i:i+self.render_chunk], cone_scale=cone_scale_gather[i:i+self.render_chunk], composite_bkgd=composite_bkgd, bkgd_color=bkgd_color, training=training)
            for key in out_chunk:
                if key not in out:
                    out[key] = []
                out[key].append(out_chunk[key])

        # Combine output, scatter and reshape to input structure
        for key in out:
            val_concat = tf.concat(out[key], 0)
            scatter_shape = [rays_o_flat.shape[0]] + list(val_concat.shape[1:])
            val_scatter = tf.scatter_nd(idxs, val_concat, shape=scatter_shape)
            # Make sure filtered rays have the correct color to not influence the loss
            if composite_bkgd and 'color' in key:
                val_scatter += tf.scatter_nd(idxs_inv, tf.broadcast_to(bkgd_color, (idxs_inv.shape[0], 3)), shape=scatter_shape)
            val_reshape = tf.reshape(val_scatter, [rays_o.shape[0], -1] + list(val_scatter.shape[1:]))
            out[key] = val_reshape

        return out

    def render_rays(self, rays_o: tf.Tensor, rays_d: tf.Tensor, t: tf.Tensor, parameters: tf.Tensor, cone_scale: tf.Tensor, composite_bkgd: bool, bkgd_color: tf.Tensor, training: bool) -> dict:
        """Ray march along given rays."""

        n_rays = rays_o.shape[0]

        # Get normalized ray direction
        rays_d_n = rays_d / tf.linalg.norm(rays_d, axis=-1, keepdims=True)

        # Get sampling locations
        t_vals = tf.linspace(0., 1., self.n_samples)
        z_vals = t[:, None, 0] * (1 - t_vals) + t[:, None, 1] * t_vals
        z_vals = tf.broadcast_to(z_vals, [n_rays, self.n_samples])

        # Jitter the sampling locations
        if self.perturb:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = tf.concat([mids, z_vals[..., -1:]], -1)
            lower = tf.concat([z_vals[..., :1], mids], -1)
            z_rand = tf.random.uniform(z_vals.shape)
            z_vals = lower + (upper - lower) * z_rand

        # Get the points correspoinding to the sampling locations
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        # Evaluate the model
        color, alpha = self.evaluate_model(pts, rays_d_n, parameters, cone_scale, z_vals, self.model, training)

        # Convert model output to meanigful values
        color_map, alpha_map, weights = self.map_model_output(color, alpha, z_vals, rays_d, composite_bkgd, bkgd_color)

        out = {'color_pred': color_map, 'alpha_pred': alpha_map}

        # Importance sample based on the weights calculated above
        if self.n_importance > 0:
            # Get sample points according to weights
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], self.n_importance, det=self.perturb)
            z_samples = tf.stop_gradient(z_samples)
            z_vals = tf.sort(tf.concat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

            # If there is an extra fine model use that, else use the normal model and evaluate it
            model_imp = self.model if self.model_fine is None else self.model_fine
            color_imp, alpha_imp = self.evaluate_model(pts, rays_d_n, parameters, cone_scale, z_vals, model_imp, training)
            color_map_imp, alpha_map_imp, _ = self.map_model_output(color_imp, alpha_imp, z_vals, rays_d, composite_bkgd, bkgd_color)

            out.update({'color_pred': color_map_imp, 'alpha_pred': alpha_map_imp, 'color_pred_coarse': color_map, 'alpha_pred_coarse': alpha_map})

        for key in out:
            tf.debugging.check_numerics(out[key], 'NaN or Inf encountered in {}.'.format(key))

        return out
        
    def evaluate_model(self, pos: tf.Tensor, dirs: tf.Tensor, parameters: tf.Tensor, cone_scale: tf.Tensor, z_vals: tf.Tensor, model: tf.keras.Model, training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
            """Split samples from each render bunch into even smaller net chunks to feed to the model."""

            # Flatten input, split it into chunks and evaluate the model there
            color = []
            alpha = []
            pos_flat = tf.reshape(pos, (-1, pos.shape[-1]))
            dirs_flat = tf.repeat(dirs, repeats=pos.shape[1], axis=0)
            n_pts = pos_flat.shape[0]
            params_flat = tf.repeat(parameters, repeats=pos.shape[1], axis=0)
            if self.blur_idx is not None:
                blur_scale = cone_scale[..., None, :] * z_vals[..., :, None]
                blur_scale_flat = tf.reshape(blur_scale, (-1, 1))
                params_flat = tf.concat([params_flat[:,:self.blur_idx], params_flat[:,self.blur_idx, None] * blur_scale_flat, params_flat[:,self.blur_idx+1:]], axis=-1)

            for i in range(0, n_pts, self.net_chunk):
                color_chunk, alpha_chunk = model((pos_flat[i:i+self.net_chunk], dirs_flat[i:i+self.net_chunk], params_flat[i:i+self.net_chunk]), training=training)
                color.append(color_chunk)
                alpha.append(alpha_chunk)

            color = tf.concat(color, 0)
            alpha = tf.concat(alpha, 0)

            return tf.reshape(color, pos.shape[:-1] + [3]), tf.reshape(alpha, pos.shape[:-1])

    def map_model_output(self, color: tf.Tensor, alpha: tf.Tensor, z_vals: tf.Tensor, rays_d: tf.Tensor, composite_bkgd: bool, bkgd_color: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Map the model output to meaningful values."""

        # Distance between evaluation positions, last step is infinity
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        #dists = tf.concat([dists, tf.broadcast_to([1e10], dists[..., :1].shape)], axis=-1)
        # Setting the last step to infinity makes sense if there is an actual background (think of an environment map), but if one's rendering with an empty background that can introduce artifacts (blowing up small density values)
        dists = tf.concat([dists, dists[..., -1:]], axis=-1)

        # Get the dist in world space, compensating for ||rays_d|| != 1
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

        if self.map_exr:
            # Map color output to > 0
            color_map = tf.nn.elu(color) + 1
        else:
            # Map color output to [0, 1]
            color_map = tf.math.sigmoid(color)

        # Add noise to density prediction for regularization
        noise = 0
        if self.raw_noise_std > 0:
            noise = tf.random.normal(alpha.shape) * self.raw_noise_std

        # Map alpha output to [0, 1]
        alpha_map = 1 - tf.exp(-tf.nn.relu(alpha + noise) * dists)
        
        # Compute weights proportional to the probability of the ray not having reflected up to the sampling point
        weights = alpha_map * tf.math.cumprod(1.-alpha_map + 1e-10, axis=-1, exclusive=True)

        # Reduce color along ray with the weights from above
        color_map = tf.reduce_sum(weights[..., None] * color_map, axis=-2)

        # Get depth map as expected distance
        depth_map = tf.reduce_sum(weights * z_vals, axis=-1)

        # Reduce weights to get the scattering probability of the ray
        alpha_map = tf.reduce_sum(weights, -1)

        # Composite on a white background
        if composite_bkgd:
            color_map = color_map + (1.-alpha_map[..., None]) * bkgd_color

        return color_map, alpha_map, weights

class InstanceRenderer(Renderer):
    """Renderer using the instancer to sample the points for ray marching the MLP."""
    # TODO: Add arguments to support the cone_scale parameter

    def __init__(self, instancer_config: EasyDict=None, step_size: float=0.002, density_scale: float=1, density_reweighting: bool=True, false_color: bool=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.instancer = util.instantiate(instancer_config)
        self.step_size = step_size
        self.density_scale = density_scale
        self.density_reweighting = density_reweighting
        self.false_color = false_color
        if false_color:
            self.instance_color = tf.random.uniform((self.instancer.n_instances(), 3))
        self.patch_scale = instancer_config['patch_scale']

    def render_rays(self, rays_o: tf.Tensor, rays_d: tf.Tensor, t: tf.Tensor, parameters: tf.Tensor, cone_scale: tf.Tensor, composite_bkgd: bool, bkgd_color: tf.Tensor, training: bool, **kwargs) -> dict:
        """Ray march along given rays."""

        assert (training == False), "network.renderer.InstanceRenderer can only be used for evaluation."
        # TODO: Add assert to enforce normalized directions

        n_rays = rays_o.shape[0]

        color_map, alpha_map = self.evaluate_model(rays_o, rays_d, parameters, cone_scale, self.model, composite_bkgd, bkgd_color, False)

        out = {'color_pred': color_map, 'alpha_pred': alpha_map}

        for key in out:
            tf.debugging.check_numerics(out[key], 'NaN or Inf encountered in {}.'.format(key))

        return out
        
    def evaluate_model(self, rays_o: tf.Tensor, rays_d: tf.Tensor, parameters: tf.Tensor, cone_scale: tf.Tensor, model: tf.keras.Model, composite_bkgd: bool, bkgd_color: tf.Tensor, training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
        """Split samples from each render bunch into even smaller net chunks to feed to the model."""

        n_rays = rays_o.shape[0]

        # Sample points only inside patch instances
        rays_d_map, pts, t, dists, color_last, alpha_last, alpha_weight, instance_id, idxs, params_map = self.instancer.get_model_input(rays_o, rays_d, parameters, self.n_samples, self.step_size)

        # No ray hit, return all zeros
        if idxs.shape[0] == 0:
            return tf.zeros((n_rays, 3), dtype=tf.float32), tf.zeros((n_rays,), dtype=tf.float32)

        # Append scaled cone_scale if self.blur_idx is not none
        if self.blur_idx is not None:
            blur_scale = cone_scale[..., None, :] * t[..., :, None] / self.patch_scale
            # print(blur_scale[0,0,0])
            params_map = tf.concat([params_map[...,:self.blur_idx], params_map[...,self.blur_idx, None] * blur_scale, params_map[...,self.blur_idx+1:]], axis=-1)

        # Cull rays that did not hit an instance
        rays_d_map = tf.gather_nd(rays_d_map, idxs)
        pts = tf.gather_nd(pts, idxs)
        dists = tf.gather_nd(dists, idxs)
        color_last = tf.gather_nd(color_last, idxs)
        alpha_last = tf.gather_nd(alpha_last, idxs)
        alpha_weight = tf.gather_nd(alpha_weight, idxs)
        instance_id = tf.gather_nd(instance_id, idxs)
        params_map = tf.gather_nd(params_map, idxs)

        # Flatten tensors, split it into chunks and evaluate the model there
        color = []
        alpha = []
        pos_flat = tf.reshape(pts, (-1, 3))
        dirs_flat = tf.reshape(rays_d_map, (-1, 3))
        params_flat = tf.reshape(params_map, (-1, params_map.shape[-1]))
        n_pts = pos_flat.shape[0]

        # Further cull points along rays that hit an instance, but lie outside of it
        idxs_pts = tf.where(tf.reshape(dists, (-1)) > 0)
        pos_flat = tf.gather_nd(pos_flat, idxs_pts)
        dirs_flat = tf.gather_nd(dirs_flat, idxs_pts)
        params_flat = tf.gather_nd(params_flat, idxs_pts)
        n_pts_within = pos_flat.shape[0]
        if n_pts_within > 0:
            for i in range(0, n_pts_within, self.net_chunk):
                color_chunk, alpha_chunk = model((pos_flat[i:i+self.net_chunk], dirs_flat[i:i+self.net_chunk], params_flat[i:i+self.net_chunk]), training=training)
                color.append(color_chunk)
                alpha.append(alpha_chunk)

            # Concatenate network output, scatter to original shape
            color = tf.scatter_nd(idxs_pts, tf.concat(color, axis=0), (n_pts,3))
            color = tf.reshape(color, pts.shape)
            alpha = tf.scatter_nd(idxs_pts, tf.concat(alpha, axis=0), (n_pts,1))
            alpha = tf.reshape(alpha, pts.shape[:-1])
            alpha *= (alpha_weight if self.density_reweighting else 1) * self.density_scale
        else:
            color = tf.zeros(pts.shape)
            alpha = tf.zeros(pts.shape[:-1])

        # Replace colors by instance colors if we render with false color
        if self.false_color:
            color = tf.reshape(tf.gather_nd(self.instance_color, tf.reshape(instance_id, (-1, 1))), color.shape)

        # Convert model output to meanigful values
        color_map, alpha_map = self.map_model_output(color, color_last, alpha, alpha_last, dists, composite_bkgd, bkgd_color)

        # Set culled rays to 0
        color_map = tf.scatter_nd(idxs, color_map, (n_rays, 3))
        alpha_map = tf.scatter_nd(idxs, alpha_map, (n_rays,))

        return color_map, alpha_map

    def map_model_output(self, color: tf.Tensor, color_last: tf.Tensor, alpha: tf.Tensor, alpha_last: tf.Tensor, dists: tf.Tensor, composite_bkgd: bool, bkgd_color: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Map the model output to meaningful values."""

        # Map color output to [0, 1] or use instance colors if we render with false color
        if self.false_color:
            color_map = tf.concat([color, color_last], axis=1)
        else:
            if self.map_exr:
                # Map color output to > 0
                color_map = tf.nn.elu(color) + 1
            else:
                # Map color output to [0, 1]
                color_map = tf.math.sigmoid(color)
            color_map = tf.concat([color_map, color_last], axis=1)

        # Add noise to density prediction for regularization
        noise = 0
        if self.raw_noise_std > 0:
            noise = tf.random.normal(alpha.shape) * self.raw_noise_std

        # Map alpha output to [0, 1]
        alpha_map = tf.concat([1 - tf.exp(-tf.nn.relu(alpha + noise) * dists / self.patch_scale), alpha_last], axis=1)
        
        # Compute weights proportional to the probability of the ray not having reflected up to the sampling point
        weights = alpha_map * tf.math.cumprod(1.-alpha_map + 1e-10, axis=-1, exclusive=True)

        # Reduce color along ray with the weights from above
        color_map = tf.reduce_sum(weights[..., None] * color_map, axis=-2)

        # Reduce weights to get the scattering probability of the ray
        alpha_map = tf.reduce_sum(weights, -1)

        # Composite on a white background
        if composite_bkgd:
            color_map = color_map + (1.-alpha_map[..., None]) * bkgd_color

        return color_map, alpha_map

class MipRenderer(Renderer):
    """Render by making use of integrated positional encodings introduced by Jon Barron et al., https://github.com/google/mipnerf."""

    def __init__(self, blur_idx: int=None, **kwargs):
        super().__init__(**kwargs)

        # Hide blur_idx from super class to avoid default filter handling
        self.blur_idx_mip = blur_idx

    def render_rays(self, rays_o: tf.Tensor, rays_d: tf.Tensor, t: tf.Tensor, parameters: tf.Tensor, cone_scale: tf.Tensor, composite_bkgd: bool, bkgd_color: tf.Tensor, training: bool) -> dict:
        """Ray march along given rays."""

        n_rays = rays_o.shape[0]

        # Get normalized ray direction
        rays_d_n = rays_d / tf.linalg.norm(rays_d, axis=-1, keepdims=True)

        # Get sampling locations
        t_vals = tf.linspace(0., 1., self.n_samples + 1)
        z_vals = t[:, None, 0] * (1 - t_vals) + t[:, None, 1] * t_vals
        z_vals = tf.broadcast_to(z_vals, [n_rays, self.n_samples + 1])

        # Jitter the sampling locations
        if self.perturb:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = tf.concat([mids, z_vals[..., -1:]], -1)
            lower = tf.concat([z_vals[..., :1], mids], -1)
            z_rand = tf.random.uniform(z_vals.shape)
            z_vals = lower + (upper - lower) * z_rand

        # Splice out the blur param
        blur = parameters[..., self.blur_idx_mip, None] * cone_scale
        parameters = tf.concat([parameters[..., :self.blur_idx_mip], parameters[..., self.blur_idx_mip+1:]], axis=-1)

        # Get the mean and variance for each cone segment
        mean, cov_diag = self.get_cone_segment_gaussians(rays_o, rays_d, z_vals, blur)
        pts = tf.concat([mean, cov_diag], axis=-1)

        # Evaluate the model
        color, alpha = self.evaluate_model(pts, rays_d_n, parameters, None, None, self.model, training)

        # Convert model output to meanigful values
        color_map, alpha_map, weights = self.map_model_output(color, alpha, z_vals, rays_d, composite_bkgd, bkgd_color)

        out = {'color_pred': color_map, 'alpha_pred': alpha_map}

        # Importance sample based on the weights calculated above
        if self.n_importance > 0:
            raise('Importance sampling for mip-NeRF style rendering is not yet implemented.')

        for key in out:
            tf.debugging.check_numerics(out[key], 'NaN or Inf encountered in {}.'.format(key))

        return out

    def get_cone_segment_gaussians(self, rays_o: tf.Tensor, rays_d: tf.Tensor, t_vals: tf.Tensor, radii: tf.Tensor):
        """Get the mean and variance approxiamting cone segments."""

        # Compute gaussian parameters with respect to the ray segments
        t0 = t_vals[..., :-1]
        t1 = t_vals[..., 1:]

        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
        t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) / (3 * mu**2 + hw**2)**2)
        r_var = radii**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 * (hw**4) / (3 * mu**2 + hw**2))

        # Transform the parameters into world space
        mean = rays_o[..., None, :] + rays_d[..., None, :] * t_mean[..., None]

        d_mag_sq = tf.maximum(1e-10, tf.reduce_sum(rays_d**2, axis=-1, keepdims=True))
        d_outer_diag = rays_d**2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag

        return mean, cov_diag

    def map_model_output(self, color: tf.Tensor, alpha: tf.Tensor, z_vals: tf.Tensor, rays_d: tf.Tensor, composite_bkgd: bool, bkgd_color: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Map the model output to meaningful values."""

        # Distance between evaluation positions, last step is infinity
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        # Get the dist in world space, compensating for ||rays_d|| != 1
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

        if self.map_exr:
            # Map color output to > 0
            color_map = tf.nn.elu(color) + 1
        else:
            # Map color output to [0, 1]
            color_map = tf.math.sigmoid(color)

        # Add noise to density prediction for regularization
        noise = 0
        if self.raw_noise_std > 0:
            noise = tf.random.normal(alpha.shape) * self.raw_noise_std

        # Map alpha output to [0, 1]
        alpha_map = 1 - tf.exp(-tf.nn.relu(alpha + noise) * dists)
        
        # Compute weights proportional to the probability of the ray not having reflected up to the sampling point
        weights = alpha_map * tf.math.cumprod(1.-alpha_map + 1e-10, axis=-1, exclusive=True)

        # Reduce color along ray with the weights from above
        color_map = tf.reduce_sum(weights[..., None] * color_map, axis=-2)

        # Reduce weights to get the scattering probability of the ray
        alpha_map = tf.reduce_sum(weights, -1)

        # Composite on a white background
        if composite_bkgd:
            color_map = color_map + (1.-alpha_map[..., None]) * bkgd_color

        return color_map, alpha_map, weights

class MipInstanceRenderer(InstanceRenderer):
    """Using integrated positional encodings with the InstanceRenderer."""

    def __init__(self, blur_idx: int=None, **kwargs):
        super().__init__(**kwargs)

        # Hide blur_idx from super class to avoid default filter handling
        self.blur_idx_mip = blur_idx

    def evaluate_model(self, rays_o: tf.Tensor, rays_d: tf.Tensor, parameters: tf.Tensor, cone_scale: tf.Tensor, model: tf.keras.Model, composite_bkgd: bool, bkgd_color: tf.Tensor, training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
        """Split samples from each render bunch into even smaller net chunks to feed to the model."""

        n_rays = rays_o.shape[0]

        # Sample points only inside patch instances
        rays_d_map, pts, t, dists, color_last, alpha_last, alpha_weight, instance_id, idxs, params_map = self.instancer.get_model_input(rays_o, rays_d, parameters, self.n_samples, self.step_size)

        # tf.debugging.Assert(tf.reduce_any(params_map >= 0.0) and tf.reduce_any(params_map <= 1.0), params_map)

        # No ray hit, return all zeros
        if idxs.shape[0] == 0:
            return tf.zeros((n_rays, 3), dtype=tf.float32), tf.zeros((n_rays,), dtype=tf.float32)

        # Cull rays that did not hit an instance
        rays_d_map = tf.gather_nd(rays_d_map, idxs)
        pts = tf.gather_nd(pts, idxs)
        t = tf.gather_nd(t, idxs)
        dists = tf.gather_nd(dists, idxs)
        color_last = tf.gather_nd(color_last, idxs)
        alpha_last = tf.gather_nd(alpha_last, idxs)
        alpha_weight = tf.gather_nd(alpha_weight, idxs)
        instance_id = tf.gather_nd(instance_id, idxs)
        params_map = tf.gather_nd(params_map, idxs)
        cone_scale = tf.gather_nd(cone_scale, idxs)

        # Splice out the blur param
        blur = params_map[..., self.blur_idx_mip] * cone_scale[..., None, 0] / self.patch_scale
        params_map = tf.concat([params_map[..., :self.blur_idx_mip], params_map[..., self.blur_idx_mip+1:]], axis=-1)

        # Flatten tensors, split it into chunks and evaluate the model there
        color = []
        alpha = []
        pos_flat = tf.reshape(pts, (-1, 3))
        dirs_flat = tf.reshape(rays_d_map, (-1, 3))
        params_flat = tf.reshape(params_map, (-1, params_map.shape[-1]))
        blur_flat = tf.reshape(blur, (-1))
        t_flat = tf.reshape(t, (-1))
        dists_flat = tf.reshape(dists, (-1))
        n_pts = pos_flat.shape[0]

        # Further cull points along rays that hit an instance, but lie outside of it
        idxs_pts = tf.where(dists_flat > 0)
        pos_flat = tf.gather_nd(pos_flat, idxs_pts)
        dirs_flat = tf.gather_nd(dirs_flat, idxs_pts)
        params_flat = tf.gather_nd(params_flat, idxs_pts)
        blur_flat = tf.gather_nd(blur_flat, idxs_pts)
        t_flat = tf.gather_nd(t_flat, idxs_pts)
        dists_flat = tf.gather_nd(dists_flat, idxs_pts)

        # Get the mean and variance for each cone segment
        mean = pos_flat
        cov_diag = self.get_cone_segment_gaussians(dirs_flat, t_flat, blur_flat, dists_flat)
        pos_flat = tf.concat([mean, cov_diag], axis=-1)

        # Call network
        n_pts_within = pos_flat.shape[0]
        if n_pts_within > 0:
            for i in range(0, n_pts_within, self.net_chunk):
                color_chunk, alpha_chunk = model((pos_flat[i:i+self.net_chunk], dirs_flat[i:i+self.net_chunk], params_flat[i:i+self.net_chunk]), training=training)
                color.append(color_chunk)
                alpha.append(alpha_chunk)

            # Concatenate network output, scatter to original shape
            color = tf.scatter_nd(idxs_pts, tf.concat(color, axis=0), (n_pts,3))
            color = tf.reshape(color, pts.shape)
            alpha = tf.scatter_nd(idxs_pts, tf.concat(alpha, axis=0), (n_pts,1))
            alpha = tf.reshape(alpha, pts.shape[:-1])
            alpha *= (alpha_weight if self.density_reweighting else 1) * self.density_scale
        else:
            color = tf.zeros(pts.shape)
            alpha = tf.zeros(pts.shape[:-1])

        # Replace colors by instance colors if we render with false color
        if self.false_color:
            color = tf.reshape(tf.gather_nd(self.instance_color, tf.reshape(instance_id, (-1, 1))), color.shape)

        # Convert model output to meanigful values
        color_map, alpha_map = self.map_model_output(color, color_last, alpha, alpha_last, dists, composite_bkgd, bkgd_color)

        # Set culled rays to 0
        color_map = tf.scatter_nd(idxs, color_map, (n_rays, 3))
        alpha_map = tf.scatter_nd(idxs, alpha_map, (n_rays,))

        return color_map, alpha_map

    def get_cone_segment_gaussians(self, rays_d: tf.Tensor, t_vals: tf.Tensor, radii: tf.Tensor, dists: tf.Tensor):
        """Get the mean and variance approxiamting cone segments."""

        # Compute gaussian parameters with respect to the ray segments
        mu = t_vals
        hw = dists
        t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) / (3 * mu**2 + hw**2)**2)
        r_var = radii**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 * (hw**4) / (3 * mu**2 + hw**2))

        # Transform the parameters into world space
        d_mag_sq = tf.maximum(1e-10, tf.reduce_sum(rays_d**2, axis=-1, keepdims=True))
        d_outer_diag = rays_d**2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[:, None] * d_outer_diag
        xy_cov_diag = r_var[:, None] * null_outer_diag
        cov_diag = t_cov_diag + xy_cov_diag

        return cov_diag

def sample_pdf(bins: tf.Tensor, weights: tf.Tensor, N_samples: int, det: bool=False) -> tf.Tensor:
    """Sample a categorical distribution given by weights, taken from the original implementation by Ben Mildenhall et al., https://github.com/bmild/nerf."""

    # Get pdf
    weights += 1e-5  # prevent nans
    pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = tf.linspace(0., 1., N_samples)
        u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = tf.random.uniform(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = tf.searchsorted(cdf, u, side='right')
    below = tf.maximum(0, inds-1)
    above = tf.minimum(cdf.shape[-1]-1, inds)
    inds_g = tf.stack([below, above], -1)
    cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples