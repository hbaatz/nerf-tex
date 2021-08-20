#include "instancer.hpp"

#include <algorithm>
#include <set>
#include <cmath>
#include <iostream>

#include <igl/readPLY.h>
#include <igl/per_vertex_normals.h>
#include <igl/principal_curvature.h>
#include <igl/avg_edge_length.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "json.hpp"

using json = nlohmann::json;

#define __forceinline inline __attribute__((always_inline))

#define MAX_TOTAL_HITS 200

inline float sign(float f) {
    return (f < 0) ? -1.f : 1.f;
}

void errorFunction(void* userPtr, enum RTCError error, const char* str)
{
    std::cout << "ERROR " << error << ": " << str << std::endl;
}

/* Read texture from file and save as a vector of matrices. */
std::vector<Eigen::MatrixXf> loadTexture(std::string texture_path) {
    std::vector<Eigen::MatrixXf> textures;

    int width, height, channels;
    unsigned char* data = stbi_load(texture_path.c_str(), &width, &height, &channels, 0);
    if (data) {
        Eigen::Map<Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> data_map(data, width * height, channels);
        Eigen::MatrixXf data_float = data_map.cast<float>() / 255.f;
        delete[] data;
        for (uint8_t channel = 0; channel < channels; ++channel) {
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> data_channel(data_float.col(channel).data(), width, height);
            textures.push_back(data_channel.rowwise().reverse());
        }
    }

    return textures;
}

/* Constructor */ 
C_Instancer::C_Instancer(std::vector<float> b_0, std::vector<float> b_1, bool cast_shadow_rays, std::vector<std::string> texture_paths, uint32_t min_shadow_samples, uint32_t n_shadow_samples, uint32_t min_texture_samples, uint32_t n_texture_samples, float jitter_amount, uint8_t instance_sample_method, bool use_mean_distance, uint32_t seed) : instancer_geomID(RTC_INVALID_GEOMETRY_ID), cast_shadow_rays(cast_shadow_rays), n_parameters(0), min_shadow_samples(min_shadow_samples), n_shadow_samples(n_shadow_samples), min_texture_samples(min_texture_samples), n_texture_samples(n_texture_samples), patch_scale(1.0), jitter_amount(jitter_amount), gen(seed), instance_sample_method(instance_sample_method), use_mean_distance(use_mean_distance) {
    /* Initialize device and set error function */
    this->device = rtcNewDevice(NULL);

    if (!this->device)
        std::cout << "ERROR " << rtcGetDeviceError(NULL) << ": cannot create device\n" << std::endl;

    rtcSetDeviceErrorFunction(device, errorFunction, NULL);

    /* Create proxy AABB that will be instanciated later on */
    this->proxy = rtcNewScene(this->device);
    rtcSetSceneFlags(this->proxy, RTC_SCENE_FLAG_ROBUST | RTC_SCENE_FLAG_CONTEXT_FILTER_FUNCTION);
    createAABB(this->device, this->proxy, b_0, b_1);
    rtcCommitScene(this->proxy);

    /* Compute maximal patch extent from origin to determin maximal point query radius */
    this->patch_max_extent = Eigen::Array3f(b_0.data()).max(Eigen::Array3f(b_1.data())).matrix().norm();

    /* Create scene to instanciate proxies into */
    this->instances = rtcNewScene(this->device);
    rtcSetSceneFlags(this->instances, RTC_SCENE_FLAG_ROBUST | RTC_SCENE_FLAG_CONTEXT_FILTER_FUNCTION);

    /* Load textures */
    for (const std::string& path : texture_paths) {
        if (path == "light") {
            this->light_dir_parameter_idx = this->n_parameters;
            this->n_parameters += 3;
        } else if (path == "point") {
            this->light_strength_parameter_idx = this->n_parameters;
            this->light_dir_parameter_idx = this->n_parameters + 1;
            this->n_parameters += 4;
        } else if (path != "") {
            std::vector<Eigen::MatrixXf> texture = loadTexture(path);
            this->textures.insert(this->textures.end(), texture.begin(), texture.end());
            this->texture_parameter_idxs.push_back(this->n_parameters);
            this->n_parameters += texture.size();
        } else {
            this->n_parameters += 1;
        }
    }
}

/* Add a template AABB of a given extent to a scene */
void C_Instancer::createAABB(RTCDevice& device, RTCScene& scene, std::vector<float> b_0, std::vector<float> b_1) {
    RTCGeometry mesh = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_QUAD);

    float* vertices = (float*) rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), 8);
    vertices[0] = b_0[0]; vertices[1] = b_0[1]; vertices[2] = b_0[2];
    vertices[3] = b_0[0]; vertices[4] = b_0[1]; vertices[5] = b_1[2]; 
    vertices[6] = b_0[0]; vertices[7] = b_1[1]; vertices[8] = b_0[2]; 
    vertices[9] = b_0[0]; vertices[10] = b_1[1]; vertices[11] = b_1[2]; 
    vertices[12] = b_1[0]; vertices[13] = b_0[1]; vertices[14] = b_0[2]; 
    vertices[15] = b_1[0]; vertices[16] = b_0[1]; vertices[17] = b_1[2]; 
    vertices[18] = b_1[0]; vertices[19] = b_1[1]; vertices[20] = b_0[2]; 
    vertices[21] = b_1[0]; vertices[22] = b_1[1]; vertices[23] = b_1[2];

    uint32_t* quads = (uint32_t*) rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT4, 4 * sizeof(uint32_t), 6);

    quads[0] = 0; quads[1] = 1; quads[2] = 3; quads[3] = 2;
    quads[4] = 0; quads[5] = 2; quads[6] = 6; quads[7] = 4;
    quads[8] = 0; quads[9] = 4; quads[10] = 5; quads[11] = 1;
    quads[12] = 7; quads[13] = 5; quads[14] = 4; quads[15] = 6;
    quads[16] = 7; quads[17] = 3; quads[18] = 1; quads[19] = 5;
    quads[20] = 7; quads[21] = 6; quads[22] = 2; quads[23] = 3;

    rtcCommitGeometry(mesh);
    this->instance_geomID = rtcAttachGeometry(scene, mesh);
    rtcReleaseGeometry(mesh);
}

/* Add AABB instance to the scene collecting all patch instances */
template <typename Container>
void C_Instancer::AddInstance(Container& transformation) {
    /* Add inverse transformation and normalized transpose to buffer */
    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> transform_mat(transformation.data());
    //std::cout << "TRAFO MAT:\n" << transform_mat << std::endl;
    this->instance_origins.push_back(transform_mat.block<3, 1>(0, 3));
    this->transformations.push_back(transform_mat.inverse());
    Eigen::Matrix3f dir_transform_mat = transform_mat.block<3,3>(0,0).transpose().rowwise().normalized();
    this->dir_transformations.push_back(dir_transform_mat);

    /* Add instance with given transformation */
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
    rtcSetGeometryInstancedScene(geom, this->proxy);
    rtcSetGeometryTransform(geom, 0, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, transform_mat.block<3, 4>(0, 0).data());
    rtcCommitGeometry(geom);
    rtcAttachGeometry(this->instances, geom);
    rtcReleaseGeometry(geom);
}

/* Data for closest point queries */
struct ClosestPointData {
    ClosestPointData(Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> & mesh_V, Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> & mesh_F) : primID(RTC_INVALID_GEOMETRY_ID), mesh_V(mesh_V), mesh_F(mesh_F) {}
    
    unsigned int primID;
    Eigen::Vector3f uvw;
    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>& mesh_V;
    Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>& mesh_F;
};

/* Find the closest point to a point p on a triangle given by vertices a, b & c */
std::tuple<Eigen::Vector3f, Eigen::Vector3f> closest_point_triangle(Eigen::Vector3f const& p, Eigen::Vector3f const& a, Eigen::Vector3f const& b, Eigen::Vector3f const& c) {
    const Eigen::Vector3f ab = b - a, ac = c - a, ap = p - a;

    /* Nearest point is a */
    const float d1 = ab.dot(ap), d2 = ac.dot(ap);
    if (d1 <= 0.f && d2 <= 0.f) return std::make_tuple(a, Eigen::Vector3f(1, 0, 0));

    /* Nearest point is b */
    const Eigen::Vector3f bp = p - b;
    const float d3 = ab.dot(bp), d4 = ac.dot(bp);
    if (d3 >= 0.f && d4 <= d3) return std::make_tuple(b, Eigen::Vector3f(0, 1, 0)); 

    /* Nearest point is c */
    const Eigen::Vector3f cp = p - c;
    const float d5 = ab.dot(cp), d6 = ac.dot(cp);
    if (d6 >= 0.f && d5 <= d6) return std::make_tuple(c, Eigen::Vector3f(0, 0, 1));

    /* Nearest point is on ab */
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
        const float v = d1 / (d1 - d3);
        return std::make_tuple(a + v * ab, Eigen::Vector3f(1 - v, v, 0));
    }

    /* Nearest point is on ac */
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
        const float v = d2 / (d2 - d6);
        return std::make_tuple(a + v * ac, Eigen::Vector3f(1 - v, 0, v));
    }

    /* Nearest point is on bc */
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
        const float v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return std::make_tuple(b + v * (c - b), Eigen::Vector3f(0, 1 - v, v));
    }

    /* Nearest point is inside the triangle */
    // TODO: One can probably stop the search if this case occurs...
    const float denom = 1.f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    return std::make_tuple(a + v * ab + w * ac, Eigen::Vector3f(1 - v - w, v, w));
}

/* A point query function finding the closest point on a mesh to a given point */
bool closest_point_query_function(struct RTCPointQueryFunctionArguments* args) {
    /* Get the primitive ID of the encountered triangle. */
    assert(args->userPtr);
    const unsigned int geomID = args->geomID;
    const unsigned int primID = args->primID;

    /* Get the position of the query point */
    const Eigen::Vector3f q(args->query->x, args->query->y, args->query->z);

    /* Get the triangle data */
    ClosestPointData* data = (ClosestPointData*)args->userPtr;
    const Eigen::Vector3i t = data->mesh_F.row(primID);
    const Eigen::Vector3f v0 = data->mesh_V.row(t(0));
    const Eigen::Vector3f v1 = data->mesh_V.row(t(1));
    const Eigen::Vector3f v2 = data->mesh_V.row(t(2));

    /* Compute the point - triangle distance */
    Eigen::Vector3f p, uvw;
    std::tie(p, uvw) = closest_point_triangle(q, v0, v1, v2);
    float d = (q - p).norm();

    if (d < args->query->radius) {
        args->query->radius = d;
        data->primID = primID;
        data->uvw = uvw;
        return true;
    }

    return false;
}

/* Load instancer mesh and distribute patch instances given a set of origins */
void C_Instancer::DistributeInstancesOnMesh(std::string mesh_path, float scale, std::string patch_origins_path) {
    /* Load mesh */
    Eigen::MatrixXi tmp_E;
    bool success = igl::readPLY(mesh_path, this->instancer_mesh.V, this->instancer_mesh.F, tmp_E, this->instancer_mesh.N, this->instancer_mesh.UV);
    if (!success) throw "ERROR: Failed reading mesh_path specified at instancer.instancer.Instancer.";

    uint32_t n_vertices = this->instancer_mesh.V.rows();
    uint32_t n_faces = this->instancer_mesh.F.rows();

    /* Save patch scale, rescale the patch extent */
    float avg_edge_length = igl::avg_edge_length(this->instancer_mesh.V, this->instancer_mesh.F);
    if (scale <= 0) scale = avg_edge_length;
    this->patch_scale = scale;
    this->patch_max_extent *= scale;

    /* Comput tangent frame */
    Eigen::MatrixXf T = Eigen::MatrixXf::Zero(n_vertices, 3), B(n_vertices, 3);
    for (uint32_t i = 0; i < n_faces; ++i) {
        Eigen::Vector3i f = this->instancer_mesh.F.row(i);
        Eigen::Vector3f e0 = this->instancer_mesh.V.row(f[1]) - this->instancer_mesh.V.row(f[0]);
        Eigen::Vector3f e1 = this->instancer_mesh.V.row(f[2]) - this->instancer_mesh.V.row(f[0]);
        Eigen::Vector2f uv0 = this->instancer_mesh.UV.row(f[1]) - this->instancer_mesh.UV.row(f[0]);
        Eigen::Vector2f uv1 = this->instancer_mesh.UV.row(f[2]) - this->instancer_mesh.UV.row(f[0]);

        float r = 1.0f / (uv0(0) * uv1(1) - uv0(1) * uv1(0));

        Eigen::Vector3f t = (e0 * uv1(1) - e1 * uv0(1)) * r;

        T.row(f(0)) += t;
        T.row(f(1)) += t;
        T.row(f(2)) += t;
    }
    for (uint32_t i = 0; i < n_vertices; ++i) {
        this->instancer_mesh.N.row(i).normalize();
        Eigen::Vector3f n = this->instancer_mesh.N.row(i);
        Eigen::Vector3f t = T.row(i);

        t = t - (n * n.dot(t));
        t.normalize();

        T.row(i) = t;
        B.row(i) = n.cross(t);
    }

    /* Load patch locations */
    Eigen::MatrixXf patch_V;
    Eigen::MatrixXi tmp_F;
    success = igl::readPLY(patch_origins_path, patch_V, tmp_F);

    /* Create patches at the locations from patch_origins_path or use the meshes vertices if none is specified */
    std::uniform_real_distribution<float> jitter_dist(0, M_PI);
    if (success) {
        /* Create a scene to find the faces corresponding to the patch origins */
        RTCGeometry mesh = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
        rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, this->instancer_mesh.V.data(), 0, 3 * sizeof(float), n_vertices);
        rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, this->instancer_mesh.F.data(), 0, 3 * sizeof(uint32_t), n_faces);

        rtcCommitGeometry(mesh);

        RTCScene fused_instancer = rtcNewScene(this->device);

        rtcAttachGeometry(fused_instancer, mesh);

        rtcReleaseGeometry(mesh);

        rtcCommitScene(fused_instancer);

        for (uint32_t i = 0; i < patch_V.rows(); ++i) {
            /* Find nearest triangle on this->instancer, interpolate principal curvatures and normal and add patch */
            Eigen::Vector3f pt = patch_V.row(i);

            RTCPointQuery query;
            query.x = pt(0);
            query.y = pt(1);
            query.z = pt(2);
            query.radius = avg_edge_length;
            query.time = 0.f;

            ClosestPointData data(this->instancer_mesh.V, this->instancer_mesh.F);
            RTCPointQueryContext context;
            rtcInitPointQueryContext(&context);
            rtcPointQuery(fused_instancer, &query, &context, closest_point_query_function, (void*)&data);

            Eigen::Vector3i f = this->instancer_mesh.F.row(data.primID);
            Eigen::Vector3f interp_N = this->instancer_mesh.N.row(f(0)) * data.uvw(0) + this->instancer_mesh.N.row(f(1)) * data.uvw(1) + this->instancer_mesh.N.row(f(2)) * data.uvw(2);

            Eigen::Vector3f interp_T = T.row(f(0)) * data.uvw(0) + T.row(f(1)) * data.uvw(1) + T.row(f(2)) * data.uvw(2);
            Eigen::Vector3f interp_B = B.row(f(0)) * data.uvw(0) + B.row(f(1)) * data.uvw(1) + B.row(f(2)) * data.uvw(2);

            interp_N.normalize();
            interp_T.normalize();
            Eigen::Vector3f interp_B_cross = interp_N.cross(interp_T);
            interp_B = interp_B_cross;

            /* Randomize patch rotation around normal */
            if (this->jitter_amount > 0) {
                float angle = this->jitter_amount * jitter_dist(this->gen);
                interp_B = interp_B * std::cos(angle) + interp_N.cross(interp_B) * std::sin(angle) + interp_N * interp_N.dot(interp_B) * (1 - std::cos(angle));
            }

            Eigen::Vector3f interp_T_cross = interp_B.cross(interp_N);
            interp_T = interp_T_cross;

            Eigen::Matrix<float, 4, 4, Eigen::RowMajor> transformation = Eigen::Matrix4f::Identity();
            transformation.block<3, 1>(0, 0) = interp_T * scale;
            transformation.block<3, 1>(0, 1) = interp_B * scale;
            transformation.block<3, 1>(0, 2) = interp_N * scale;
            transformation.block<3, 1>(0, 3) = patch_V.row(i);
            this->AddInstance(transformation);
        }

        rtcReleaseScene(fused_instancer);
    } else {
        std::vector<Eigen::Matrix<float, 1, 3>> tmp_V;
        for (uint32_t i = 0; i < n_vertices; ++i) {
            auto it = std::find(tmp_V.begin(), tmp_V.end(), this->instancer_mesh.V.row(i));
            if (it != tmp_V.end()) continue;

            Eigen::Vector3f tmp_T = T.row(i), tmp_B = B.row(i), tmp_N = this->instancer_mesh.N.row(i);
            /* Randomize patch rotation around normal */
            if (this->jitter_amount > 0) {
                float angle = this->jitter_amount * jitter_dist(this->gen);
                tmp_B = tmp_B * std::cos(angle) + tmp_N.cross(tmp_B) * std::sin(angle) + tmp_N * tmp_N.dot(tmp_B) * (1 - std::cos(angle));
                Eigen::Vector3f tmp_T_cross = tmp_N.cross(tmp_B);
                tmp_T = sign(tmp_T.dot(tmp_T_cross)) * tmp_T_cross;
            }

            Eigen::Matrix<float, 4, 4, Eigen::RowMajor> transformation = Eigen::Matrix4f::Identity();
            transformation.block<3, 1>(0, 0) = tmp_T * scale;
            transformation.block<3, 1>(0, 1) = tmp_B * scale;
            transformation.block<3, 1>(0, 2) = tmp_N * scale;
            transformation.block<3, 1>(0, 3) = this->instancer_mesh.V.row(i);
            this->AddInstance(transformation);
            tmp_V.push_back(this->instancer_mesh.V.row(i));
        }
    }

    /* Add the mesh itself to both its own scene (for texture coordinate calculation) and the scene shared with the patch bbox instances (for culling) */
    n_vertices = this->instancer_mesh.V.rows();
    n_faces = this->instancer_mesh.F.rows();
    RTCGeometry mesh = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
    rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, this->instancer_mesh.V.data(), 0, 3 * sizeof(float), n_vertices);
    rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, this->instancer_mesh.F.data(), 0, 3 * sizeof(uint32_t), n_faces);

    rtcCommitGeometry(mesh);

    this->instancer_geomID = rtcAttachGeometry(this->instances, mesh);
    this->max_geomID = this->instancer_geomID;
    this->instancer_mesh.geomID = this->instancer_geomID;

    this->instancer = rtcNewScene(this->device);
    rtcAttachGeometry(this->instancer, mesh);

    rtcReleaseGeometry(mesh);

    /* Commit instancer, the instances are commited manually later */
    rtcCommitScene(this->instancer);
}

/* Add auxilliary mesh to the this->instances scene */
void C_Instancer::AddMesh(std::string mesh_path, std::string texture_path) {
    /* Load mesh */
    unsigned int mesh_geomID = ++this->max_geomID;
    this->meshes[mesh_geomID] = Mesh();
    Mesh& mesh_data = this->meshes[mesh_geomID];

    Eigen::MatrixXi tmp_E;
    bool success = igl::readPLY(mesh_path, mesh_data.V, mesh_data.F, tmp_E, mesh_data.N, mesh_data.UV);
    if (!success) throw "ERROR: Failed reading mesh_path specified at instancer.instancer.Instancer.";

    /* Load texture */
    mesh_data.textures = loadTexture(texture_path);

    /* Add mesh to the scene */
    uint32_t n_vertices = mesh_data.V.rows();
    uint32_t n_faces = mesh_data.F.rows();

    RTCGeometry mesh = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
    rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, mesh_data.V.data(), 0, 3 * sizeof(float), n_vertices);
    rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, mesh_data.F.data(), 0, 3 * sizeof(uint32_t), n_faces);

    rtcCommitGeometry(mesh);
    rtcAttachGeometryByID(this->instances, mesh, mesh_geomID);
    rtcReleaseGeometry(mesh);
}

/* Commit the this->instances scene */
void C_Instancer::CommitScene() {
    /* Commit scene with added instances */
    rtcCommitScene(this->instances);
}

/* Return the number of instances, used for false coloring in nertwork.renderer */
uint32_t C_Instancer::GetNumberOfInstances() {
    return this->transformations.size();
}

/* Extended ray structure that gathers all hits along the ray, see Embree's Next Hit tutorial */
struct HitList
{
HitList (unsigned int instance_geomID = -1)
    : begin(0), end(0), instance_geomID(instance_geomID) {}

    /* Hit structure that defines complete order over hits */
    struct Hit {
        Hit() {}

        Hit (float t,  unsigned int primID = 0xFFFFFFFF, unsigned int geomID = 0xFFFFFFFF, unsigned int instID = 0xFFFFFFFF, float u = 0, float v = 0)
        : t(t), primID(primID), geomID(geomID), instID(instID), u(u), v(v) {}

        /* lexicographical order (t,instID,geomID,primID) */
        __forceinline friend bool operator < (const Hit& a, const Hit& b) {
            if (a.t == b.t) {
                if (a.instID == b.instID) {
                    if (a.geomID == b.geomID) return a.primID < b.primID;
                    else                      return a.geomID < b.geomID;
                }
                else return a.instID < b.instID;
            }
            return a.t < b.t;
        }

        __forceinline friend bool operator == (const Hit& a, const Hit& b) {
            return a.t == b.t && a.primID == b.primID && a.geomID == b.geomID && a.instID == b.instID;
        }

        __forceinline friend bool operator <= (const Hit& a, const Hit& b) {
            if (a == b) return true;
            else return a < b;
        }

        __forceinline friend bool operator != (const Hit& a, const Hit& b) {
            return !(a == b);
        }

    public:
        float t;
        unsigned int primID;
        unsigned int geomID;
        unsigned int instID;
        float u;
        float v;
    };

    /* return number of gathered hits */
    unsigned int size() const {
        return end-begin;
    }

    /* returns the last hit */
    const Hit& last() const {
        assert(end);
        return hits[end-1];
    }

    public:
    unsigned int begin;             // begin of hit list
    unsigned int end;               // end of hit list
    Hit hits[MAX_TOTAL_HITS];       // array to store all found hits to
    unsigned int instance_geomID;   // geomID of the instances bboxes, used to check wether an instance or another mesh was hit
};

/* Data for primary ray hit queries */
struct IntersectContext {
    IntersectContext(HitList& hits) : hits(hits) {}

    RTCIntersectContext context;
    HitList& hits;
};

/* Data for shadow ray hit queries */
struct ShadowContext {
    ShadowContext(unsigned int instance_geomID) : instance_geomID(instance_geomID) {}

    RTCIntersectContext context;
    unsigned int instance_geomID;
};

/* Ray ininitalization utility */
inline void init_ray(RTCRay& ray, const Eigen::Vector3f& org, const Eigen::Vector3f& dir, float tnear, float tfar, unsigned int mask, unsigned int flags) {
    ray.org_x = org(0);
    ray.org_y = org(1);
    ray.org_z = org(2);
    ray.dir_x = dir(0);
    ray.dir_y = dir(1);
    ray.dir_z = dir(2);
    ray.tnear = tnear;
    ray.tfar = tfar;
    ray.mask = mask;
    ray.flags = flags;
}

/* Primary ray hit filter function, used to collect hits and cull when possible */
void primary_ray_filter_function(const struct RTCFilterFunctionNArguments* args) {
    assert(*args->valid == -1);
    IntersectContext* context = (IntersectContext*) args->context;
    HitList& hits = context->hits;
    RTCRay* ray = (RTCRay*) args->ray;
    RTCHit* hit = (RTCHit*) args->hit;
    assert(args->N == 1);
    if (hit->geomID == hits.instance_geomID) args->valid[0] = 0; // ignore all hits on instances

    /* avoid overflow of hits array */
    if (hits.end >= MAX_TOTAL_HITS) return;
    
    /* add hit to list, ignore in order duplicates */
    HitList::Hit hit_0 = HitList::Hit(ray->tfar, hit->primID, hit->geomID, hit->instID[0], hit->u, hit->v);
    if (hits.end == 0 || hits.hits[hits.end-1] != hit_0) hits.hits[hits.end++] = hit_0;
}

/* Shadow ray hit filter function, used to check if other patch instances or auxilliary meshes were hit */
void shadow_ray_filter_function(const struct RTCFilterFunctionNArguments* args) {
    assert(*args->valid == -1);
    ShadowContext* context = (ShadowContext*) args->context;
    RTCRay* ray = (RTCRay*) args->ray;
    RTCHit* hit = (RTCHit*) args->hit;
    assert(args->N == 1);

    /* Check if the ray hit either another patch bbox from above, an instancer or auxilliary mesh or the bottom of a patch bbox */
    float hit_outside = ray->dir_x * hit->Ng_x + ray->dir_y * hit->Ng_y + ray->dir_z * hit->Ng_z < 0;
    if (!((hit->primID == 4 && hit_outside) || (hit->geomID != context->instance_geomID && hit_outside) || hit->primID == 1)) args->valid[0] = 0;
}

/* Map point from world to patch local coordinates */
inline Eigen::Vector3f C_Instancer::getPt(const Eigen::Vector3f& pt, unsigned int instID) {
    return this->transformations[instID].block<3, 3>(0, 0) * pt + this->transformations[instID].block<3, 1>(0, 3);
}

/* Map direction from world to patch local coordinates */
inline Eigen::Vector3f C_Instancer::getDir(const Eigen::Vector3f& dir, unsigned int instID) {
    return this->dir_transformations[instID] * dir.normalized();
}

/* Get world coordinates of point along a ray given by its origin and direction */
inline Eigen::Vector3f C_Instancer::getPtOnRay(const Eigen::Vector3f& org, const Eigen::Vector3f& dir, float t) {
    return org + t * dir;
}

/* Return patch local light direction or [0, 0, -1] if shadowed (this corresponds to ambient lighting only) */
inline Eigen::Vector3f C_Instancer::getShadowedLightDir(bool isShadowed, const Eigen::Vector3f& dir, const Eigen::Vector3f& pt, unsigned int instID) {
    if (isShadowed) {
        return Eigen::Vector3f(0, 0, -1);
    } else {
        if (this->light_strength_parameter_idx >= 0) {
            return this->getDir(dir - pt, instID);
        } else {
            return this->getDir(dir, instID);
        }
    }
}

/* Return the radiance arriving from a point light at a point in world space */
float C_Instancer::getLightStrength(float lightStrength, const Eigen::Vector3f& lightPos, const Eigen::Vector3f& pt) {
    float eps = 1e-6;
    float distSquared = (lightPos - pt).squaredNorm();

    return lightStrength / (4 * M_PI * distSquared + eps);
}

/* Execute a occlusion query to check if point pt is shadowed in direction dir */
bool C_Instancer::isShadowed(const Eigen::Vector3f& pt, const Eigen::VectorXf& dir) {
    ShadowContext shadow_context(this->instance_geomID);
    rtcInitIntersectContext(&shadow_context.context);
    shadow_context.context.filter = shadow_ray_filter_function;
    RTCRay shadow_ray;
    init_ray(shadow_ray, pt, dir, 0, 100, -1, 0);
    rtcOccluded1(this->instances, &shadow_context.context, &shadow_ray);

    return shadow_ray.tfar < 0;
}

/* Interpolate y_ref at position x */
float C_Instancer::interpolate2d(Eigen::Vector2f x, Eigen::MatrixXf& y_ref) {
    /* Scale x up to the pixel grid */
    x = x.array() * Eigen::Array2f(y_ref.rows() - 1.f, y_ref.cols() - 1.f);

    /* Get pixel indices of x */
    Eigen::Vector2i idx_00 = x.cast<int>();
    Eigen::Vector2i idx_01 = x.cast<int>() + Eigen::Vector2i(0, 1);
    Eigen::Vector2i idx_10 = x.cast<int>() + Eigen::Vector2i(1, 0);
    Eigen::Vector2i idx_11 = x.cast<int>() + Eigen::Vector2i(1, 1);

    assert(idx_11(0) < y_ref.rows() && idx_11(1) < y_ref.cols());

    Eigen::Vector2f weights = x.array() - x.array().floor();

    float y_interp = y_ref(idx_00(0), idx_00(1))  * (1 - weights(0)) * (1 - weights(1))
                     + y_ref(idx_01(0), idx_01(1)) * (1 - weights(0)) * weights(1)
                     + y_ref(idx_10(0), idx_10(1)) * weights(0) * (1 - weights(1))
                     + y_ref(idx_11(0), idx_11(1)) * weights(0) * weights(1);
                     
    return y_interp;
}

/* Interpolate a set of textures y_ref at x */
Eigen::VectorXf C_Instancer::interpolate2d(Eigen::Vector2f x, std::vector<Eigen::MatrixXf>& y_ref) {
    uint32_t channels = y_ref.size();
    Eigen::VectorXf y_interp(channels);

    for (uint8_t channel = 0; channel < channels; ++channel) {
        y_interp(channel) = this->interpolate2d(x, y_ref[channel]);
    }
                     
    return y_interp;
}

/* Get material parameters at point pt by looking up the texture at the nearest point of the instancer mesh */
Eigen::VectorXf C_Instancer::getParameters(const Eigen::Vector3f& pt, const Eigen::VectorXf& parameters) {
    Eigen::VectorXf parameter_map = parameters;

    /* Find nearest point on instancer mesh */
    RTCPointQuery query;
    query.x = pt(0);
    query.y = pt(1);
    query.z = pt(2);
    query.radius = this->patch_max_extent;
    query.time = 0.f;

    ClosestPointData data(this->instancer_mesh.V, this->instancer_mesh.F);
    RTCPointQueryContext context;
    rtcInitPointQueryContext(&context);
    rtcPointQuery(this->instancer, &query, &context, closest_point_query_function, (void*)&data);

    for (uint32_t i = 0; i < texture_parameter_idxs.size(); ++i) {
        if (data.primID != RTC_INVALID_GEOMETRY_ID) {
            assert(data.primID < this->instancer_mesh.F.rows() && data.primID != RTC_INVALID_GEOMETRY_ID);
            Eigen::Vector3i f = this->instancer_mesh.F.row(data.primID);
            assert(f(0) < this->instancer_mesh.UV.rows() && f(2) < this->instancer_mesh.UV.rows() && f(2) < this->instancer_mesh.UV.rows());
            Eigen::Vector2f uv = this->instancer_mesh.UV.row(f(0)) * data.uvw(0) + this->instancer_mesh.UV.row(f(1)) * data.uvw(1) + this->instancer_mesh.UV.row(f(2)) * data.uvw(2);
            parameter_map(texture_parameter_idxs[i]) *= this->interpolate2d(uv, this->textures[i]);
        }
    }

    return parameter_map;
}

/* Auxilliary function to select a random path instance if a sample point lies within multiple */
template <typename Iterator>
inline std::tuple<unsigned int, float> C_Instancer::sampleRandom(Iterator begin, Iterator end) {
    uint32_t n_elements = std::distance(begin, end);
    std::uniform_int_distribution<> dis(0, n_elements - 1);
    std::advance(begin, dis(this->gen));
    return std::make_tuple(*begin, (float) n_elements);
}

/* Auxilliary function to select the nearest path instance if a sample point lies within multiple */
template <typename Iterator>
inline std::tuple<unsigned int, float> C_Instancer::sampleNearest(Iterator begin, Iterator end, const Eigen::Vector3f& pt) {
    float min_dist = std::numeric_limits<float>::infinity();
    for (Iterator it = begin; it != end; ++it) {
        unsigned int instID = *it;
        float dist = (pt - this->instance_origins[instID]).norm();
        if (dist < min_dist) {
            begin = it;
            min_dist = dist;
        }
    }
    return std::make_tuple(*begin, 1.f);
}

/* Auxilliary function to blend between the nearest path instances if a sample point lies within multiple */
template <typename Iterator>
inline std::tuple<unsigned int, float> C_Instancer::sampleNearestBlend(Iterator begin, Iterator end, const Eigen::Vector3f& pt, float transition_range) {
    transition_range *= this->patch_scale;
    float min_dist = std::numeric_limits<float>::infinity();
    std::vector<float> weights;
    for (Iterator it = begin; it != end; ++it) {
        float dist = (pt - this->instance_origins[*it]).norm();
        weights.push_back(dist);
        if (dist < min_dist) {
            min_dist = dist;
        }
    }
    for (auto& w : weights) {
        w = std::max(transition_range + min_dist - w, 0.f);
    }
    std::discrete_distribution<uint32_t> dis(weights.begin(), weights.end());
    uint32_t idx = dis(this->gen);
    std::advance(begin, idx);
    return std::make_tuple(*begin, 1 / dis.probabilities()[idx]);
}

/* Dummy shading model for auxilliary meshes */
Eigen::Vector3f C_Instancer::shadeMesh(const Eigen::Vector3f& pt, unsigned int geomID, unsigned int primID, const Eigen::Vector3f& uvw, const Eigen::Vector3f& dir, float diffuse, float ambient) {
    /* Get mesh given the geomID */
    Mesh& mesh = this->meshes[geomID];

    /* Compute surface normal */
    Eigen::Vector3i f = mesh.F.row(primID);
    Eigen::Vector3f n = mesh.N.row(f(0)) * uvw(0) + mesh.N.row(f(1)) * uvw(1) + mesh.N.row(f(2)) * uvw(2);
    n.normalize();

    /* Get texture value */
    Eigen::VectorXf albedo;
    if (mesh.textures.empty()) {
        albedo = Eigen::Vector3f(.8,.8,.8);
    } else {
        Eigen::Vector2f uv = mesh.UV.row(f(0)) * uvw(0) + mesh.UV.row(f(1)) * uvw(1) + mesh.UV.row(f(2)) * uvw(2);
        albedo = this->interpolate2d(uv, mesh.textures);
        if (albedo.size() != 3) albedo = Eigen::Vector3f::Constant(albedo(0));
    }

    /* Compute diffuse lighting contribution */
    if (!this->isShadowed(pt + n * 1e-6, dir)) {
        diffuse *= std::max(n.dot(dir.normalized()), 0.f);
    } else {
        diffuse = 0;
    }

    return albedo * std::min(diffuse + ambient, 1.f);
}

/* Auxilliary function to get the mean distance of a cone segment used for IPEs */
float C_Instancer::getMeanDistance(float mu, float hw) {
    return mu + 2 * mu * std::pow(hw, 2) / (3 * std::pow(mu, 2) + std::pow(hw, 2));
}

/* Ray march the scene with a set of given rays and and map the points, directions and material parameters according to the patch instances the sample points fall into */
void C_Instancer::GetModelInput(float* rays_o, float* rays_d, float* t, float* dists, float* pts, float* color, float* density, float* density_weight, int* instance_id, bool* hit, uint32_t n_rays, uint32_t n_pts, float step_size, float* parameters) {
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> rays_o_map(rays_o, n_rays, 3);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> rays_d_map(rays_d, n_rays * n_pts, 3);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> pts_map(pts, n_rays * n_pts, 3);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> parameters_map(parameters, n_rays * n_pts, this->n_parameters);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> color_map(color, n_rays, 3);

    /* Get all resources neccessary for tracing a ray */
    HitList hits(this->instance_geomID);
    IntersectContext context(hits);
    rtcInitIntersectContext(&context.context);
    context.context.filter = primary_ray_filter_function;
    RTCRayHit rayhit;
    std::set<unsigned int> active_instances;

    /* Distribution to sample random offset along the ray to shift the sample points */
    std::uniform_real_distribution<float> offset_dist(0, 1);

    /* Collect some debug info */
    uint32_t min_buffer_size = n_pts;

    for (uint32_t i = 0; i < n_rays; ++i) {
        /* Set up ray struct */
        Eigen::Vector3f default_raydir = rays_d_map.row(n_pts*i);
        init_ray(rayhit.ray, rays_o_map.row(i), default_raydir, 0, 100, -1, 0);
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

        /* Trace the rays and record all intersections */
        rtcIntersect1(this->instances, &context.context, &rayhit);

        /* Skip if no intersection was found */
        if (!hits.size()) continue;

        hit[i] = true;

        /* sort hits by extended order */
        std::sort(&hits.hits[hits.begin], &hits.hits[hits.end]);

        /* ignore out of order duplicates */
        if (hits.size())
        {
            uint32_t i = 0, j = 1;
            for (; j < hits.size(); j++) {
                if (hits.hits[i] == hits.hits[j]) continue;
                hits.hits[++i] = hits.hits[j];
            }
            hits.end = i + 1;
        }

        /* Sum up ray segment length inside instances */
        std::vector<float> segment_lengths;
        float total_segment_length = 0, t_entry = 0;
        HitList::Hit* mesh_hit = nullptr;
        for (uint32_t j = hits.begin; j < hits.end; ++j) {
            if (hits.hits[j].geomID != instance_geomID) {
                if (!active_instances.empty()) {
                    float segment_length = hits.hits[j].t - t_entry;
                    total_segment_length += segment_length;
                    segment_lengths.push_back(segment_length);
                }
                mesh_hit = &hits.hits[j];
                break;
            }
            unsigned int instID = hits.hits[j].instID;
            if (active_instances.find(instID) != active_instances.end()) {
                active_instances.erase(instID);
                if (active_instances.empty()) {
                    float segment_length = hits.hits[j].t - t_entry;
                    total_segment_length += segment_length;
                    segment_lengths.push_back(segment_length);
                }
            } else {
                if (active_instances.empty()) t_entry = hits.hits[j].t;
                active_instances.insert(instID);
            }
        }
        
        /* Intermediate clean up */
        active_instances.clear();

        /* Get the light direction (or point light position) */
        Eigen::Vector3f default_lightdir;
        if (this->light_dir_parameter_idx >= 0) {
            default_lightdir = parameters_map.block<1, 3>(n_pts*i, this->light_dir_parameter_idx);
        }
        float default_lightstr;
        if (this->light_strength_parameter_idx >= 0) {
            default_lightstr = parameters_map(n_pts*i, this->light_strength_parameter_idx);
        }

        if (total_segment_length > 0) {
            /* Find the number of steps to cover the segments and get random offset along the ray to place the samples */
            uint32_t neccessary_steps = (uint32_t) (total_segment_length / step_size);
            uint32_t n_steps = std::min(neccessary_steps, n_pts);
            float t_offset = 0;
            if (n_steps == 0) {
                dists[n_pts*i] = total_segment_length;
                t_offset = offset_dist(this->gen) * total_segment_length;
                n_steps = 1;
            } else {
                if (n_steps < neccessary_steps) min_buffer_size = std::max(min_buffer_size, neccessary_steps);
                for (uint32_t j = n_pts*i, k = 0; j < n_pts*i + n_steps - 1; ++j) {
                    dists[j] = step_size;
                }
                dists[n_pts*i + n_steps - 1] = step_size + total_segment_length - n_steps * step_size;

                t_offset = offset_dist(this->gen) * step_size;
            }

            /* Get the number of shadow and texture samples */
            uint32_t n_shadow_samples = std::max(this->min_shadow_samples, (uint32_t) (this->n_shadow_samples * total_segment_length));
            uint32_t n_texture_samples = std::max(this->min_texture_samples, (uint32_t) (this->n_texture_samples * total_segment_length));

            /* Loop trough intersections and set pts & parameters accordingly */
            float segment_offset = 0, cleared_segment_length = 0; t_entry = 0;
            uint32_t k_shadow, k_texture;
            float t_0_shadow, t_1_shadow, t_0_texture, t_1_texture, step_length_shadow, step_length_texture;
            float sample_0_shadow, sample_1_shadow;
            Eigen::VectorXf sample_0_texture, sample_1_texture, default_parameters = parameters_map.row(n_pts*i);
            for (uint32_t j = hits.begin, step = 0, l = 0; j < hits.end && step < n_steps; ++j) {
                /* Get t_pt such that o + d * t_pt = pt */
                float t_mu = step * step_size + t_offset + segment_offset, t_pt;
                if (this->use_mean_distance) {
                    t_pt = this->getMeanDistance(t_mu, step_size);
                } else {
                    t_pt = t_mu;
                }

                while (!active_instances.empty() && (t_pt < hits.hits[j].t) && step < n_steps) {
                    uint32_t k = n_pts*i + step;

                    /* Save t for later */
                    t[k] = t_mu;

                    /* Get the world coordinates of the point corresponding to t_pt */
                    Eigen::Vector3f pt = getPtOnRay(rays_o_map.row(i), default_raydir, t_pt);

                    /* If the point is inside multiple patch instances, sample one */
                    unsigned int instID;
                    if (active_instances.size() == 1) {
                        instID = *active_instances.begin();
                        density_weight[k] = 1.0;
                        instance_id[k] = (int) instID;
                    } else {
                        switch(this->instance_sample_method) {
                            case 0:
                                std::tie(instID, density_weight[k]) = this->sampleRandom(active_instances.begin(), active_instances.end());
                                break;
                            case 1:
                                std::tie(instID, density_weight[k]) = this->sampleNearest(active_instances.begin(), active_instances.end(), pt);
                                break;
                            case 2:
                                std::tie(instID, density_weight[k]) = this->sampleNearestBlend(active_instances.begin(), active_instances.end(), pt);
                                break;
                        }
                        instance_id[k] = (int) instID;
                    }

                    /* Map parameters given a texure on the instancer mesh */
                    if (this->instancer_geomID != RTC_INVALID_GEOMETRY_ID && !this->texture_parameter_idxs.empty() && n_texture_samples < n_pts) {
                        /* Interpolate parameters */
                        /* Update parameter interpolation values */
                        while (t_pt > t_1_texture) {
                            t_0_texture = t_1_texture;
                            t_1_texture = t_entry + ++k_texture * step_length_texture;
                            sample_0_texture = sample_1_texture;
                            sample_1_texture = this->getParameters(this->getPtOnRay(rays_o_map.row(i), default_raydir, t_1_texture), default_parameters);
                        }

                        /* Interpolate parameter values */
                        float w = (t_pt - t_0_texture) / step_length_texture; // Linear interpolation
                        parameters_map.row(k) = sample_0_texture * (1 - w) + sample_1_texture * w;
                    } else if (this->instancer_geomID != RTC_INVALID_GEOMETRY_ID && !this->texture_parameter_idxs.empty()) {
                        /* Query parameters at actual point, no interpolation */
                        parameters_map.row(k) = this->getParameters(pt, default_parameters);
                    }

                    /* Map the light direction to the local patch coordinates. */
                    if (this->light_dir_parameter_idx >= 0) {
                        if (this->cast_shadow_rays && n_shadow_samples < n_pts) {
                            /* Interpolate shadow */
                            /* Update shadow interpolation values */    
                            while (t_pt > t_1_shadow) {
                                t_0_shadow = t_1_shadow;
                                t_1_shadow = t_entry + ++k_shadow * step_length_shadow;
                                sample_0_shadow = sample_1_shadow;
                                sample_1_shadow = this->isShadowed(this->getPtOnRay(rays_o_map.row(i), default_raydir, t_1_shadow), default_lightdir);
                            }

                            /* Interpolate shadow values */
                            bool w = (t_pt - t_0_shadow) / step_length_shadow >= 0.5f; // Nearest neighbor interpolation
                            parameters_map.block<1, 3>(k, this->light_dir_parameter_idx) = this->getShadowedLightDir(!w && sample_0_shadow || w && sample_1_shadow, default_lightdir, pt, instID);
                        } else if (this->cast_shadow_rays) {
                            /* Query shadow at actual point, no interpolation */
                            parameters_map.block<1, 3>(k, this->light_dir_parameter_idx) = this->getShadowedLightDir(this->isShadowed(pt, default_lightdir), default_lightdir, pt, instID);
                        } else {
                            /* No self shadowing */
                            parameters_map.block<1, 3>(k, this->light_dir_parameter_idx) = this->getShadowedLightDir(false, default_lightdir, pt, instID);
                        }
                    }

                    /* Map the light strength if the scene contains a point light */
                    if (this->light_strength_parameter_idx >= 0) {
                        parameters_map(k, this->light_strength_parameter_idx) = this->getLightStrength(default_lightstr, default_lightdir, pt);
                    }

                    /* Map the point in world coordinates to the local patch coordinates */
                    pts_map.row(k) = this->getPt(pt, instID);
                    rays_d_map.row(k) = this->getDir(default_raydir, instID);

                    /* update t_pt */
                    ++step;

                    t_mu = step * step_size + t_offset + segment_offset;
                    if (this->use_mean_distance) {
                        t_pt = this->getMeanDistance(t_mu, step_size);
                    } else {
                        t_pt = t_mu;
                    }
                }
                if (hits.hits[j].geomID != this->instance_geomID) break; // cull remaining patch instances if ray hit instancer
                /* Keep track of the instances we're currently inside in */
                unsigned int instID = hits.hits[j].instID;
                if (active_instances.find(instID) != active_instances.end()) {
                    active_instances.erase(instID);

                    /* End of segment */
                    if (active_instances.empty()){
                        cleared_segment_length += hits.hits[j].t - t_entry;
                    }
                } else {
                    /* Beginning of segment */
                    if (active_instances.empty()) {
                        segment_offset = hits.hits[j].t - cleared_segment_length;
                        t_entry = hits.hits[j].t;

                        /* Initialize shadow and parameter interpolation values */
                        if (this->instancer_geomID != RTC_INVALID_GEOMETRY_ID && !this->texture_parameter_idxs.empty() && n_texture_samples < n_pts) {
                            float segment_length = segment_lengths[l];
                            uint32_t n_texture_samples_segment = std::max(this->min_texture_samples, (uint32_t)(n_texture_samples * segment_length / total_segment_length));
                            step_length_texture = segment_length / (n_texture_samples_segment - 1);
                            k_texture = 1;
                            t_0_texture = t_entry;
                            t_1_texture = t_entry + step_length_texture;
                            sample_0_texture = this->getParameters(this->getPtOnRay(rays_o_map.row(i), default_raydir, t_0_texture), default_parameters);
                            sample_1_texture = this->getParameters(this->getPtOnRay(rays_o_map.row(i), default_raydir, t_1_texture), default_parameters);
                        }

                        if (this->light_dir_parameter_idx >= 0 && this->cast_shadow_rays && n_shadow_samples < n_pts) {
                            float segment_length = segment_lengths[l];
                            uint32_t n_shadow_samples_segment = std::max(this->min_shadow_samples, (uint32_t) (n_shadow_samples * segment_length / total_segment_length));
                            step_length_shadow = segment_length / (n_shadow_samples_segment - 1);
                            k_shadow = 1;
                            t_0_shadow = t_entry;
                            t_1_shadow = t_entry + step_length_shadow;
                            sample_0_shadow = this->isShadowed(this->getPtOnRay(rays_o_map.row(i), default_raydir, t_0_shadow), default_lightdir);
                            sample_1_shadow = this->isShadowed(this->getPtOnRay(rays_o_map.row(i), default_raydir, t_1_shadow), default_lightdir);
                        }

                        ++l;
                    }
                    active_instances.insert(instID);
                }
            }
        }

        /* Set final sample point color to black if hit instancer mesh, to the color of the provided texture if it hit an auxilliary mesh and else ignore, i.e. set density to 0 */
        if (mesh_hit) {
            if (mesh_hit->geomID == this->instancer_geomID) {
                color_map.row(i) = Eigen::Vector3f::Zero();
            } else {
                color_map.row(i) = this->shadeMesh(getPtOnRay(rays_o_map.row(i), default_raydir, mesh_hit->t), mesh_hit->geomID, mesh_hit->primID, Eigen::Vector3f(1 - mesh_hit->u - mesh_hit->v, mesh_hit->u, mesh_hit->v), default_lightdir);
            }
            density[i] = 1;
        } else {
            color_map.row(i) = Eigen::Vector3f::Zero();
            density[i] = 0;
        }

        /* Clean up */
        hits.begin = 0; hits.end = 0;
        active_instances.clear();
    }

    if (min_buffer_size > n_pts) std::cout << "WARNING: BUFFER SIZE TO SMALL, NEED " << min_buffer_size << ", GOT " << n_pts << "\n";
}

/* Export the patch instance transformations as json file */
void C_Instancer::ExportTransformations(std::string file_path) {
    /* Invert the transformations in this->transformations and export as json file. */
    json root = json::array();

    for (Eigen::Matrix4f& trafo : this->transformations) {
        Eigen::Matrix4f trafo_inv = trafo.inverse();
        json mat = json::array();
        for (uint8_t i = 0; i < 4; i++) {
            json row = json::array();
            for (uint8_t j = 0; j < 4; j++) {
                row.push_back(trafo_inv(i, j));
            }
            mat.push_back(row);
        }
        root.push_back(mat);
    }

    std::ofstream out(file_path);
    out << root.dump(4);
    out.close();
    std::cout << file_path << std::endl;
}
