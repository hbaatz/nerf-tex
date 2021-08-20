from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "src/instancer.cpp":
    pass

cdef extern from "src/instancer.hpp":
    cdef cppclass C_Instancer:
        C_Instancer(vector[float] b_0, vector[float] b_1, bool cast_shadow_rays, vector[string] texture_paths, int min_shadow_samples, int n_shadow_samples, int min_texture_samples, int n_texture_samples, float jitter_amount, int instance_sample_method, bool use_mean_distance) except +
        void AddInstance(vector[float] transformation)
        void DistributeInstancesOnMesh(string mesh_path, float scale, string patch_origins_path)
        void AddMesh(string mesh_path, string texture_path)
        void CommitScene()
        int GetNumberOfInstances();
        void GetModelInput(float* rays_o, float* rays_d, float* t, float* dists, float* pts, float* color, float* density, float* density_weight, int* instance_id, bool* hit, int n_rays, int n_pts, float step_size, float* parameters)
        void ExportTransformations(string file_path)