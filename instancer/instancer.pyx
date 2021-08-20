from cython.operator cimport dereference as deref
cimport numpy as np
import numpy as np
import tensorflow as tf

cdef class Instancer:
    cdef C_Instancer* thisptr

    def __cinit__(self, list b_0, list b_1, cast_shadow_rays=False, list textures=[], list transformations=[], str mesh_path=None, float patch_scale=1., patch_origins_path='', int min_shadow_samples=4, int n_shadow_samples=512, int min_texture_samples=4, int n_texture_samples=512, float jitter_amount=0, str instance_sampling_method='random', bool use_mean_distance=False, auxiliary_meshes=[], transformation_export_path=None):
        textures_encoded = map(lambda t: t.encode('utf-8'), textures)
        
        instance_sample_method = {'random': 0, 'nearest': 1, 'nearest_blend': 2}[instance_sampling_method]

        self.thisptr = new C_Instancer(b_0, b_1, cast_shadow_rays, textures_encoded, min_shadow_samples, n_shadow_samples, min_texture_samples, n_texture_samples, jitter_amount, instance_sample_method, use_mean_distance)

        for transformation in transformations:
            self.thisptr.AddInstance(sum(transformation, []))

        if mesh_path is not None:
            self.thisptr.DistributeInstancesOnMesh(mesh_path.encode('utf-8'), patch_scale, patch_origins_path.encode('utf-8'))
            if transformation_export_path is not None:
                self.thisptr.ExportTransformations(transformation_export_path.encode('utf-8'))

        for aux_mesh_path, aux_texture_path in auxiliary_meshes:
            self.thisptr.AddMesh(aux_mesh_path.encode('utf-8'), aux_texture_path.encode('utf-8'))

        self.thisptr.CommitScene()
    
    def n_instances(self):
        return self.thisptr.GetNumberOfInstances()

    def __dealloc__(self):
        del self.thisptr

    def get_model_input_np(self, np.ndarray[float, ndim=2, mode="c"] rays_o, np.ndarray[float, ndim=3, mode="c"] rays_d, np.ndarray[float, ndim=2, mode="c"] t, np.ndarray[float, ndim=2, mode="c"] dist, np.ndarray[float, ndim=3, mode="c"] pts, np.ndarray[float, ndim=3, mode="c"] color, np.ndarray[float, ndim=2, mode="c"] density, np.ndarray[float, ndim=2, mode="c"] density_weight, np.ndarray[int, ndim=2, mode="c"] instance_id, np.ndarray[bool, ndim=1, mode="c"] hit, float step_size, np.ndarray[float, ndim=3, mode="c"] parameters):
            self.thisptr.GetModelInput(&rays_o[0,0], &rays_d[0,0,0], &t[0,0], &dist[0,0], &pts[0,0,0], &color[0,0,0], &density[0,0], &density_weight[0,0], &instance_id[0,0], &hit[0], pts.shape[0], pts.shape[1], step_size, &parameters[0,0,0])

    def get_model_input(self, rays_o, rays_d, parameters, n_samples, step_size):
        n_rays = rays_o.shape[0]

        rays_d_np = np.repeat(rays_d.numpy()[:,None,:], n_samples, axis=1)
        t = np.zeros((n_rays, n_samples), dtype=np.float32)
        dist = np.zeros((n_rays, n_samples), dtype=np.float32)
        pts_np = np.zeros((n_rays, n_samples, 3), dtype=np.float32)
        color = np.zeros((n_rays, 1, 3), dtype=np.float32)
        density = np.zeros((n_rays, 1), dtype=np.float32)
        density_weight = np.ones((n_rays, n_samples), dtype=np.float32)
        instance_id = np.zeros((n_rays, n_samples), dtype=np.int32)
        hit = np.zeros(n_rays, dtype=np.bool)
        parameters_np = np.repeat(parameters[:, None, :], repeats=n_samples, axis=1)

        self.get_model_input_np(rays_o.numpy(), rays_d_np, t, dist, pts_np, color, density, density_weight, instance_id, hit, step_size, parameters_np)

        return tf.constant(rays_d_np), tf.constant(pts_np), tf.constant(t), tf.constant(dist), tf.constant(color), tf.constant(density), tf.constant(density_weight), tf.constant(instance_id), tf.where(hit), tf.constant(parameters_np)