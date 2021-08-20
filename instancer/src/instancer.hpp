#ifndef PROXY_H
#define PROXY_H

#include <vector>
#include <map>
#include <random>
#include <Eigen/Dense>
#include <embree3/rtcore.h>

class C_Instancer {
    public:
        C_Instancer(std::vector<float> b_0, std::vector<float> b_1, bool cast_shadow_rays = false, std::vector<std::string> texture_paths = {}, uint32_t min_shadow_samples = 4, uint32_t n_shadow_samples = 512, uint32_t min_texture_samples = 4, uint32_t n_texture_samples = 512, float jitter_amount = 0, uint8_t instance_sample_method = false, bool use_mean_distance = false, uint32_t seed = 0);
        template <typename Container>
        void AddInstance(Container& transformation);
        void DistributeInstancesOnMesh(std::string mesh_path, float scale, std::string patch_origins_path = "");
        void AddMesh(std::string mesh_path, std::string texture_path = "");
        void CommitScene();
        uint32_t GetNumberOfInstances();
        void GetModelInput(float* rays_o, float* rays_d, float* t, float* dist, float* pts, float* color, float* density, float* density_weight, int* instance_id, bool* hit, uint32_t n_rays, uint32_t n_pts, float step_size, float* parameters = nullptr);
        void ExportTransformations(std::string file_path);

    private:

        struct Mesh {
            unsigned int geomID;
            Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> V;
            Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> F;
            Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> N;
            Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor> UV;
            std::vector<Eigen::MatrixXf> textures;
        };

        std::map<unsigned int, Mesh> meshes;

        RTCDevice device;
        RTCScene proxy;
        RTCScene instances;
        RTCScene instancer;
        unsigned int instance_geomID;
        unsigned int instancer_geomID;
        unsigned int max_geomID;
        Mesh instancer_mesh;
        std::vector<Eigen::Vector3f> instance_origins;
        std::vector<Eigen::Matrix4f> transformations;
        std::vector<Eigen::Matrix3f> dir_transformations;
        float patch_scale;
        float patch_max_extent;
        float jitter_amount;
        
        bool use_mean_distance;

        bool cast_shadow_rays;
        uint32_t min_shadow_samples;
        uint32_t n_shadow_samples;

        uint32_t n_parameters;
        int32_t light_dir_parameter_idx = -1;
        int32_t light_strength_parameter_idx = -1;
        std::vector<uint32_t> texture_parameter_idxs;
        std::vector<Eigen::MatrixXf> textures;
        uint32_t min_texture_samples;
        uint32_t n_texture_samples;

        void createAABB(RTCDevice& device, RTCScene& scene, std::vector<float> b_0, std::vector<float> b_1);

        Eigen::Vector3f getPt(const Eigen::Vector3f& pt, unsigned int instID);
        Eigen::Vector3f getDir(const Eigen::Vector3f& dir, unsigned int instID);
        Eigen::Vector3f getPtOnRay(const Eigen::Vector3f& org, const Eigen::Vector3f& dir, float t);
        Eigen::Vector3f getShadowedLightDir(bool isShadowed, const Eigen::Vector3f& dir, const Eigen::Vector3f& pt, unsigned int instID);
        float getLightStrength(float lightStrength, const Eigen::Vector3f& lightPos, const Eigen::Vector3f& pt);
        bool isShadowed(const Eigen::Vector3f& pt, const Eigen::VectorXf& dir);
        Eigen::VectorXf getParameters(const Eigen::Vector3f& pt, const Eigen::VectorXf& parameters);
        float interpolate2d(Eigen::Vector2f x, Eigen::MatrixXf& y_ref);
        Eigen::VectorXf interpolate2d(Eigen::Vector2f x, std::vector<Eigen::MatrixXf>& y_ref);

        std::mt19937 gen;
        uint8_t instance_sample_method;
        
        template <typename Iterator>
        std::tuple<unsigned int, float> sampleRandom(Iterator begin, Iterator end);
        template <typename Iterator>
        std::tuple<unsigned int, float> sampleNearest(Iterator begin, Iterator end, const Eigen::Vector3f& pt);
        template <typename Iterator>
        std::tuple<unsigned int, float> sampleNearestBlend(Iterator begin, Iterator end, const Eigen::Vector3f& pt, float transition_range = 0.2f);

        float getMeanDistance(float mu, float hw);

        Eigen::Vector3f shadeMesh(const Eigen::Vector3f& pt, unsigned int geomID, unsigned int primID, const Eigen::Vector3f& uvw, const Eigen::Vector3f& dir, float diffuse = 1.0, float ambient = 0.2);
};

#endif