#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/sdf.h>
#include <drjit/tensor.h>
#include <drjit/texture.h>

#if defined(MI_ENABLE_CUDA)
    #include "optix/sdfgrid.cuh"
#endif

NAMESPACE_BEGIN(mitsuba)

/**!

.. _shape-sdfgrid:

SDF Grid (:monosp:`sdfgrid`)
-------------------------------------------------

Props:
 * "normals": type of normal computation method (analytic, smoth, falcao)
 * "watertight": if the SDF should be watertight (default: true)

Documentation notes:
 * Grid position [0, 0, 0] x [1, 1, 1]
 * Reminder that tensors use [Z, Y, X, C] indexing
 * Does not emit UVs for texturing
 * Cannot be used for area emitters
 * Grid data must be initialized by using `mi.traverse()` (by default the plugin
   is initialized with a 2x2x2 grid of minus ones)

 Temorary issues:
     * Embree does not work

//TODO: Test that instancing works
*/

template <typename Float, typename Spectrum>
class SDFGrid final : public SDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(SDF, m_to_world, m_to_object, m_is_instance, initialize,
                   mark_dirty, get_children_string, parameters_grad_enabled)
    MI_IMPORT_TYPES()

    using typename Base::ScalarSize;

    SDFGrid(const Properties &props) : Base(props) {
        std::string normals_mode_str = props.string("normals", "smooth");
        if (normals_mode_str == "analytic")
            m_normal_method = Analytic;
        else if (normals_mode_str == "smooth")
            m_normal_method = Smooth;
        else if (normals_mode_str == "falcao")
            m_normal_method = Falcao;
        else
            Throw("Invalid normals mode \"%s\", must be one of: \"analytic\", "
                  "\"smooth\" or \"falcao\"!", normals_mode_str);

        m_watertight = props.get<bool>("watertight", true);

        std::string interpolation_mode_str = props.string("interpolation", "linear");
        if (interpolation_mode_str == "cubic") {
            m_interpolation = Cubic;
            NotImplementedError("Soon"); //FIXME: remove
        } else if (interpolation_mode_str == "linear")
            m_interpolation = Linear;
        else
            Throw("Invalid interpolation mode \"%s\", must be one of: "
                  "\"linear\" or \"cubic\"!", interpolation_mode_str);

        float grid_data[8] = { -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f };
        size_t shape[4] = {2, 2, 2, 1};
        TensorXf grid = TensorXf(grid_data, 4, shape);
        m_grid_texture = Texture3f(grid, true, false, dr::FilterMode::Linear,
                                   dr::WrapMode::Clamp);

        update();
        initialize();
    }

    ~SDFGrid() {
#if defined(MI_ENABLE_CUDA)
        jit_free(m_optix_bboxes);
        jit_free(m_optix_voxel_indices);
#endif
    }

    void update() {
        auto [S, Q, T] = dr::transform_decompose(m_to_world.scalar().matrix, 25);
        if (dr::abs(Q[0]) > 1e-6f || dr::abs(Q[1]) > 1e-6f ||
            dr::abs(Q[2]) > 1e-6f || dr::abs(Q[3] - 1) > 1e-6f)
            Log(Warn, "'to_world' transform shouldn't perform any rotations, "
                      "use instancing (`shapegroup` and `instance` plugins) "
                      "instead!");

        m_to_object = m_to_world.value().inverse();
        mark_dirty();
   }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_parameter("to_world", *m_to_world.ptr(), ParamFlags::Differentiable | ParamFlags::Discontinuous);
        callback->put_parameter("grid", m_grid_texture.tensor(), ParamFlags::Differentiable | ParamFlags::Discontinuous);
        callback->put_parameter("watertight", m_watertight, +ParamFlags::NonDifferentiable);
    }

    void parameters_changed(const std::vector<std::string> &keys) override {
        if (keys.empty() || string::contains(keys, "to_world")  || string::contains(keys, "grid")) {
            // Ensure previous ray-tracing operation are fully evaluated before
            // modifying the scalar values of the fields in this class
            if constexpr (dr::is_jit_v<Float>)
                dr::sync_thread();

            // Update the scalar value of the matrix
            m_to_world = m_to_world.value();

            m_grid_texture.set_tensor(m_grid_texture.tensor());
            update();
        }

        Base::parameters_changed();
    }

    ScalarSize primitive_count() const override {
        return m_filled_voxel_count;
    }

    ScalarBoundingBox3f bbox() const override {
        ScalarBoundingBox3f bbox;
        ScalarTransform4f to_world = m_to_world.scalar();

        bbox.expand(to_world.transform_affine(ScalarPoint3f(0.f, 0.f, 0.f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f(1.f, 0.f, 0.f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f(0.f, 1.f, 0.f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f(1.f, 1.f, 0.f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f(0.f, 0.f, 1.f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f(1.f, 0.f, 1.f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f(0.f, 1.f, 1.f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f(1.f, 1.f, 1.f)));

        return bbox;
    }

    Float surface_area() const override {
        // TODO: area emitter
        return 0;
    }

    // =============================================================
    //! @{ \name Sampling routines
    // =============================================================

    PositionSample3f sample_position(Float time, const Point2f &sample,
                                     Mask active) const override {
        // TODO: area emitter
        MI_MASK_ARGUMENT(active);
        (void) time;
        (void) sample;
        PositionSample3f ps = dr::zeros<PositionSample3f>();
        return ps;
    }

    Float pdf_position(const PositionSample3f & /*ps*/, Mask active) const override {
        // TODO: area emitter
        MI_MASK_ARGUMENT(active);
        return 0;
    }


    SurfaceInteraction3f eval_parameterization(const Point2f &uv,
                                               uint32_t ray_flags,
                                               Mask active) const override {
        // TODO: area emitter
        MI_MASK_ARGUMENT(active);
        (void) uv;
        (void) ray_flags;

        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        return si;
    }

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

    template <typename FloatP, typename Ray3fP>
    std::tuple<FloatP, Point<FloatP, 2>, dr::uint32_array_t<FloatP>,
               dr::uint32_array_t<FloatP>>
    ray_intersect_preliminary_impl(const Ray3fP &ray_,
                                   dr::mask_t<FloatP> active) const {
        MI_MASK_ARGUMENT(active);
        (void) ray_;
        // TODO: embree || differentiable
        return { dr::select(active, 0, dr::Infinity<FloatP>),
                 Point<FloatP, 2>(0, 0), ((uint32_t) -1), 0 };
    }

    template <typename FloatP, typename Ray3fP>
    dr::mask_t<FloatP> ray_test_impl(const Ray3fP &ray_,
                                     dr::mask_t<FloatP> active) const {
        MI_MASK_ARGUMENT(active);
        (void) ray_;
        // TODO: embree || differentiable
        return active;
    }

    MI_SHAPE_DEFINE_RAY_INTERSECT_METHODS()

    SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                     const PreliminaryIntersection3f &pi,
                                                     uint32_t ray_flags,
                                                     uint32_t recursion_depth,
                                                     Mask active) const override {
        MI_MASK_ARGUMENT(active);
        constexpr bool IsDiff = dr::is_diff_v<Float>;

        // Early exit when tracing isn't necessary
        if (!m_is_instance && recursion_depth > 0)
            return dr::zeros<SurfaceInteraction3f>();

        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();

        bool detach_shape = has_flag(ray_flags, RayFlags::DetachShape);
        bool follow_shape = has_flag(ray_flags, RayFlags::FollowShape);

        Transform4f to_world = m_to_world.value();
        Transform4f to_object = m_to_object.value();

        // TODO: Make sure this is the proper way to detach dr::Texture objects
        dr::suspend_grad<Float> scope(detach_shape, to_world, to_object, m_grid_texture.tensor().array());

        if constexpr (IsDiff) {
            if (follow_shape) {
                /* FollowShape glues the interaction point with the shape.
                   Therefore, to also account for a possible differential motion
                   of the shape, we first compute a detached intersection point
                   in local space and transform it back in world space to get a
                   point rigidly attached to the shape's motion, including
                   translation, scaling and rotation. */
                Point3f local_p = dr::detach(to_object.transform_affine(ray(pi.t)));
                Vector3f local_grad = dr::detach(sdf_grad(local_p));
                Normal3f local_n = dr::normalize(local_grad);
                Ray3f local_ray = dr::detach(to_object.transform_affine(ray));

                /* Note: Only when applying a motion to the entire shape is the
                 * interaction point truly "glued" to the shape. For a single
                 * voxel, the motion of the surface is ambiguous and therefore
                 * the interaction point is not "glued" to the shape. */

                // Capture gradients of `m_grid_texture`
                Float sdf_value;
                m_grid_texture.eval(rescale_point(local_p), &sdf_value);
                Point3f local_motion = sdf_value * (-local_n) / dr::dot(local_n, local_grad);
                local_p = dr::replace_grad(local_p, local_motion);

                // Capture gradients of `m_to_world`
                si.p = to_world.transform_affine(local_p);
                si.t = dr::sqrt(dr::squared_norm(si.p - ray.o) / dr::squared_norm(ray.d));
            } else {
                /* To ensure that the differential interaction point stays along
                   the traced ray, we first recompute the intersection distance
                   in a differentiable way (w.r.t. to the grid parameters) and
                   then compute the corresponding point along the ray. (Instead
                   of computing an intersection with the SDF, we compute an
                   intersection with the tangent plane.) */
                Point3f local_p = dr::detach(to_object.transform_affine(ray(pi.t)));
                Ray3f local_ray = dr::detach(to_object.transform_affine(ray));

                /// Differntiable tangent plane normal
                // Capture gradients of `m_grid_texture`
                Normal3f local_n = dr::normalize(sdf_grad(local_p));
                // Capture gradients of `m_to_world`
                Normal3f n = to_world.transform_affine(local_n);

                /// Differentiable tangent plane point
                // Capture gradients of `m_grid_texture`
                Float sdf_value;
                m_grid_texture.eval(rescale_point(local_p), &sdf_value);
                Float t_diff = sdf_value / dr::dot(dr::detach(local_n), -local_ray.d);
                t_diff = dr::replace_grad(pi.t, t_diff);
                // Capture gradients of `m_to_world`
                Point3f p = to_world.transform_affine(local_ray(t_diff));

                si.t = dr::dot(p - ray.o, n) / dr::dot(n, ray.d);
                si.p = ray(si.t);
            }
        } else {
            si.t = pi.t;
            si.p = ray(si.t);
        }

        si.t = dr::select(active, si.t, dr::Infinity<Float>);

        Vector3f grad = sdf_grad(m_to_object.value().transform_affine(si.p));
        si.n = dr::normalize(m_to_world.value().transform_affine(Normal3f(grad)));

        if (likely(has_flag(ray_flags, RayFlags::ShadingFrame))) {
            switch (m_normal_method) {
                case Analytic:
                    si.sh_frame.n = si.n;
                    break;
                case Smooth:
                    si.sh_frame.n = smooth(m_to_object.value().transform_affine(si.p));
                    break;
                case Falcao:
                    si.sh_frame.n = falcao(m_to_object.value().transform_affine(si.p));
                    break;
                default:
                    Throw("Unknown normal computation.");
            }
        }

        si.uv = Point2f(0.f, 0.f);
        si.dp_du = Vector3f(0.f);
        si.dp_dv = Vector3f(0.f);
        si.dn_du = si.dn_dv = dr::zeros<Vector3f>();

        si.shape    = this;
        si.instance = nullptr;

        if (unlikely(has_flag(ray_flags, RayFlags::BoundaryTest))) {
            Float dp = dr::dot(si.sh_frame.n, -ray.d);
            // Add non-linearity by squaring the returned value
            si.boundary_test = dr::sqr(dp);
        }

        return si;
    }

    Normal3f smooth_sh(const Point3f &p, const Float *u_ptr, const Float *v_ptr,
                       const Float *w_ptr) const override {
        /**
           Herman Hansson-Söderlund, Alex Evans, and Tomas Akenine-Möller, Ray
           Tracing of Signed Distance Function Grids, Journal of Computer
           Graphics Techniques (JCGT), vol. 11, no. 3, 94-113, 2022
        */
        auto shape = m_grid_texture.tensor().shape();
        Vector3f resolution = Vector3f(shape[2] - 1, shape[1] - 1, shape[0] - 1);
        Point3f scaled_p = p * resolution;

        Point3i v000 = Point3i(round(scaled_p)) + Vector3i(-1, -1, -1);
        Point3i v100 = v000 + Vector3i(1, 0, 0);
        Point3i v010 = v000 + Vector3i(0, 1, 0);
        Point3i v110 = v000 + Vector3i(1, 1, 0);
        Point3i v001 = v000 + Vector3i(0, 0, 1);
        Point3i v101 = v000 + Vector3i(1, 0, 1);
        Point3i v011 = v000 + Vector3i(0, 1, 1);
        Point3i v111 = v000 + Vector3i(1, 1, 1);

        // Detect voxels that are outside of the grid, their normals will not
        // be used in the interpolation
        Bool s000 = !dr::any(v000 < 0);
        Bool s100 = !dr::any(v100 < 0);
        Bool s010 = !dr::any(v010 < 0);
        Bool s110 = !dr::any(v110 < 0);
        Bool s001 = !dr::any(v001 < 0);
        Bool s101 = !dr::any(v101 < 0);
        Bool s011 = !dr::any(v011 < 0);
        Bool s111 = !dr::any(v111 < 0);

        Vector3f n000 = dr::select(s000, dr::normalize(voxel_grad(p, v000)), Vector3f(0.f));
        Vector3f n100 = dr::select(s100, dr::normalize(voxel_grad(p, v100)), Vector3f(0.f));
        Vector3f n010 = dr::select(s010, dr::normalize(voxel_grad(p, v010)), Vector3f(0.f));
        Vector3f n110 = dr::select(s110, dr::normalize(voxel_grad(p, v110)), Vector3f(0.f));
        Vector3f n001 = dr::select(s001, dr::normalize(voxel_grad(p, v001)), Vector3f(0.f));
        Vector3f n101 = dr::select(s101, dr::normalize(voxel_grad(p, v101)), Vector3f(0.f));
        Vector3f n011 = dr::select(s011, dr::normalize(voxel_grad(p, v011)), Vector3f(0.f));
        Vector3f n111 = dr::select(s111, dr::normalize(voxel_grad(p, v111)), Vector3f(0.f));

        Vector3f diff = scaled_p - Vector3f(v111) + Vector3f(0.5);
        Float& u = diff[0];
        Float& v = diff[1];
        Float& w = diff[2];
        if (u_ptr)
            u = *u_ptr;
        if (v_ptr)
            v = *v_ptr;
        if (w_ptr)
            w = *w_ptr;

        // Disable weighting on invalid axis
        Bool invalid_x_0 = !s000 && !s010 && !s001 && !s011;
        Bool invalid_x_1 = !s100 && !s110 && !s101 && !s111;
        Bool invalid_y_0 = !s000 && !s100 && !s001 && !s101;
        Bool invalid_y_1 = !s010 && !s110 && !s011 && !s111;
        Bool invalid_z_0 = !s000 && !s100 && !s010 && !s110;
        Bool invalid_z_1 = !s001 && !s101 && !s011 && !s111;

        u = dr::select(invalid_x_0, 1, u);
        u = dr::select(invalid_x_1, 0, u);
        v = dr::select(invalid_y_0, 1, v);
        v = dr::select(invalid_y_1, 0, v);
        w = dr::select(invalid_z_0, 1, w);
        w = dr::select(invalid_z_1, 0, w);

        Normal3f n =
            (1 - w) * ((1 - v) * ((1 - u) * n000 + u * n100) + v * ((1 - u) * n010 + u * n110)) +
                  w * ((1 - v) * ((1 - u) * n001 + u * n101) + v * ((1 - u) * n011 + u * n111));

        return n;
    };

    Normal3f smooth(const Point3f &p) const override {
        Normal3f n = smooth_sh(p, nullptr, nullptr, nullptr);
        return dr::normalize(m_to_world.value().transform_affine(Normal3f(n)));
    }

    Matrix3f smooth_hessian(const Point3f& p) const override {
        Float one = 1;
        Float zero = 0;

        Normal3f n1vw = smooth_sh(p, &one,    nullptr, nullptr);
        Normal3f n0vw = smooth_sh(p, &zero,   nullptr, nullptr);
        Normal3f nu1w = smooth_sh(p, nullptr, &one,    nullptr);
        Normal3f nu0w = smooth_sh(p, nullptr, &zero,   nullptr);
        Normal3f nuv1 = smooth_sh(p, nullptr, nullptr, &one);
        Normal3f nuv0 = smooth_sh(p, nullptr, nullptr, &zero);

        Normal3f dx = n1vw - n0vw;
        Normal3f dy = nu1w - nu0w;
        Normal3f dz = nuv1 - nuv0;

        Matrix3f hessian;
        hessian(0, 0) = dx[0];
        hessian(0, 1) = dx[1];
        hessian(0, 2) = dx[2];
        hessian(1, 0) = dy[0];
        hessian(1, 1) = dy[1];
        hessian(1, 2) = dy[2];
        hessian(2, 0) = dz[0];
        hessian(2, 1) = dz[1];
        hessian(2, 2) = dz[2];

        return hessian; // normalizazion and to_world transform ?
    }

    bool parameters_grad_enabled() const override {
        return dr::grad_enabled(m_to_world);
    }

#if defined(MI_ENABLE_CUDA)
    using Base::m_optix_data_ptr;

    void optix_prepare_geometry() override {
        if constexpr (dr::is_cuda_v<Float>) {
            // TODO: more efficient memory allocations
            if (!m_optix_data_ptr)
                m_optix_data_ptr = jit_malloc(AllocType::Device, sizeof(OptixSDFGridData));
            if (m_optix_bboxes)
                jit_free(m_optix_bboxes);
            if (m_optix_voxel_indices)
                jit_free(m_optix_voxel_indices);

            std::tie(m_optix_bboxes, m_optix_voxel_indices, m_filled_voxel_count) = build_bboxes();
            if (m_filled_voxel_count == 0)
                Throw("SDFGrid should at least have one non-empty voxel!");

            size_t resolution[3] = { m_grid_texture.tensor().shape()[2],
                                     m_grid_texture.tensor().shape()[1],
                                     m_grid_texture.tensor().shape()[0] };

            OptixSDFGridData data = { (size_t*) m_optix_voxel_indices,
                                      resolution[0],
                                      resolution[1],
                                      resolution[2],
                                      m_grid_texture.tensor().array().data(),
                                      m_to_object.scalar(),
                                      m_watertight
            };
            jit_memcpy(JitBackend::CUDA, m_optix_data_ptr, &data, sizeof(OptixSDFGridData));
        }
    }

    void optix_build_input(OptixBuildInput &build_input) const override {
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build_input.customPrimitiveArray.aabbBuffers   = &m_optix_bboxes;
        build_input.customPrimitiveArray.numPrimitives = m_filled_voxel_count;
        build_input.customPrimitiveArray.strideInBytes = 6 * sizeof(float);
        build_input.customPrimitiveArray.flags         = optix_geometry_flags;
        build_input.customPrimitiveArray.numSbtRecords = 1;
    }
#endif

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SDFgrid[" << std::endl
            << "  to_world = " << string::indent(m_to_world, 13) << "," << std::endl
            << "  " << string::indent(get_children_string()) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    /* \brief Offsets and rescales an point in [0, 1] x [0, 1] x [0, 1] to
     * its corresponding point in the texture. This is usually necessary because
     * dr::Texture objects assume that the value of a pixel is positionned in
     * the middle of the pixel. For a 3D grid, this means that values are not
     * at the corners, but in the middle of the voxels.
     */
    MI_INLINE Point3f rescale_point(const Point3f& p) const {
        auto shape = m_grid_texture.tensor().shape();
        // TODO: save inv_shape to memory?
        Vector3f inv_shape(1.f / shape[2], 1.f / shape[1], 1.f / shape[0]);
        return p * (1 - inv_shape) + (inv_shape / 2.f);
    }

    /* \brief Only computes AABBs for voxel that contain a surface in it.
     * Returns a device pointer to the array of AABBs, a device pointer to
     * an array of voxel indices of the former AABBs and the count of voxels with
     * surface in them.
     */
    std::tuple<void*, void*, size_t> build_bboxes() {
        auto shape = m_grid_texture.tensor().shape();
        size_t shape_v[3]  = { shape[2], shape[1], shape[0] };
        float shape_rcp[3] = { 1.f / (shape[0] - 1), 1.f / (shape[1] - 1), 1.f / (shape[2] - 1) };
        size_t max_voxel_count = (shape[0] - 1) * (shape[1] - 1) * (shape[2] - 1);
        ScalarTransform4f to_world = m_to_world.scalar();

        float *grid = (float *) jit_malloc_migrate(
            m_grid_texture.tensor().array().data(), AllocType::Host, false);
        jit_sync_thread();

        size_t count = 0;
        optix::BoundingBox3f* aabbs = new optix::BoundingBox3f[max_voxel_count]();
        size_t* voxel_indices = new size_t[max_voxel_count]();
        for (size_t z = 0; z < shape[0] - 1; ++z) {
            for (size_t y = 0; y < shape[1] - 1 ; ++y) {
                for (size_t x = 0; x < shape[2] - 1; ++x) {
                    size_t v000 = (x + 0) + (y + 0) * shape_v[0] + (z + 0) * shape_v[0] * shape_v[1];
                    size_t v100 = (x + 1) + (y + 0) * shape_v[0] + (z + 0) * shape_v[0] * shape_v[1];
                    size_t v010 = (x + 0) + (y + 1) * shape_v[0] + (z + 0) * shape_v[0] * shape_v[1];
                    size_t v110 = (x + 1) + (y + 1) * shape_v[0] + (z + 0) * shape_v[0] * shape_v[1];
                    size_t v001 = (x + 0) + (y + 0) * shape_v[0] + (z + 1) * shape_v[0] * shape_v[1];
                    size_t v101 = (x + 1) + (y + 0) * shape_v[0] + (z + 1) * shape_v[0] * shape_v[1];
                    size_t v011 = (x + 0) + (y + 1) * shape_v[0] + (z + 1) * shape_v[0] * shape_v[1];
                    size_t v111 = (x + 1) + (y + 1) * shape_v[0] + (z + 1) * shape_v[0] * shape_v[1];

                    float f000 = grid[v000];
                    float f100 = grid[v100];
                    float f010 = grid[v010];
                    float f110 = grid[v110];
                    float f001 = grid[v001];
                    float f101 = grid[v101];
                    float f011 = grid[v011];
                    float f111 = grid[v111];

                    // No surface within voxel
                    if (f000 > 0 && f100 > 0 && f010 > 0 && f110 > 0 &&
                        f001 > 0 && f101 > 0 && f011 > 0 && f111 > 0)
                        continue;

                    ScalarBoundingBox3f bbox;
                    bbox.expand(to_world.transform_affine(ScalarPoint3f(
                        (x + 0) * shape_rcp[2], (y + 0) * shape_rcp[1], (z + 0) * shape_rcp[0])));
                    bbox.expand(to_world.transform_affine(ScalarPoint3f(
                        (x + 1) * shape_rcp[2], (y + 0) * shape_rcp[1], (z + 0) * shape_rcp[0])));
                    bbox.expand(to_world.transform_affine(ScalarPoint3f(
                        (x + 0) * shape_rcp[2], (y + 1) * shape_rcp[1], (z + 0) * shape_rcp[0])));
                    bbox.expand(to_world.transform_affine(ScalarPoint3f(
                        (x + 1) * shape_rcp[2], (y + 1) * shape_rcp[1], (z + 0) * shape_rcp[0])));
                    bbox.expand(to_world.transform_affine(ScalarPoint3f(
                        (x + 1) * shape_rcp[2], (y + 0) * shape_rcp[1], (z + 1) * shape_rcp[0])));
                    bbox.expand(to_world.transform_affine(ScalarPoint3f(
                        (x + 1) * shape_rcp[2], (y + 0) * shape_rcp[1], (z + 1) * shape_rcp[0])));
                    bbox.expand(to_world.transform_affine(ScalarPoint3f(
                        (x + 0) * shape_rcp[2], (y + 1) * shape_rcp[1], (z + 1) * shape_rcp[0])));
                    bbox.expand(to_world.transform_affine(ScalarPoint3f(
                        (x + 1) * shape_rcp[2], (y + 1) * shape_rcp[1], (z + 1) * shape_rcp[0])));

                    size_t voxel_index =
                        (x + 0) + (y + 0) * (shape_v[0] - 1) +
                        (z + 0) * (shape_v[0] - 1) * (shape_v[1] - 1);
                    voxel_indices[count] = voxel_index;
                    aabbs[count] = optix::BoundingBox3f(bbox);

                    count++;
                }
            }
        }

        jit_free(grid);

        //TODO: async memcpy
        void *aabbs_ptr = jit_malloc(AllocType::Device, sizeof(optix::BoundingBox3f) * count);
        jit_memcpy(JitBackend::CUDA, aabbs_ptr, aabbs, sizeof(optix::BoundingBox3f) * count);

        void *voxel_indices_ptr = jit_malloc(AllocType::Device, sizeof(size_t) * count);
        jit_memcpy(JitBackend::CUDA, voxel_indices_ptr, voxel_indices, sizeof(size_t) * count);

        return {aabbs_ptr, voxel_indices_ptr, count};
    }

    /// Computes the gradient for a specific gradient
    Vector3f voxel_grad(const Point3f& p, const Point3i& voxel_index) const {
        auto shape = m_grid_texture.tensor().shape();
        Vector3f resolution = Vector3f(shape[2] - 1, shape[1] - 1, shape[0] - 1);

        Float f[6];
        Point3f query;

        Point3f voxel_size = 1.f / resolution;
        Point3f p000 = Point3f(voxel_index) * voxel_size;

        query = rescale_point(Point3f(p000[0] + voxel_size[0], p[1], p[2]));
        m_grid_texture.eval(query, &f[0]);
        query = rescale_point(Point3f(p000[0], p[1], p[2]));
        m_grid_texture.eval(query, &f[1]);

        query = rescale_point(Point3f(p[0], p000[1] + voxel_size[1], p[2]));
        m_grid_texture.eval(query, &f[2]);
        query = rescale_point(Point3f(p[0], p000[1], p[2]));
        m_grid_texture.eval(query, &f[3]);

        query = rescale_point(Point3f(p[0], p[1], p000[2] + voxel_size[2]));
        m_grid_texture.eval(query, &f[4]);
        query = rescale_point(Point3f(p[0], p[1], p000[2] ));
        m_grid_texture.eval(query, &f[5]);

        Float dx = f[0] - f[1]; // f(1, y, z) - f(0, y, z)
        Float dy = f[2] - f[3]; // f(x, 1, z) - f(x, 0, z)
        Float dz = f[4] - f[5]; // f(x, y, 1) - f(x, y, 0)

        return Vector3f(dx, dy, dz);
    }

    Vector3f sdf_grad(const Point3f& p) const {
        auto shape = m_grid_texture.tensor().shape();
        Vector3f resolution = Vector3f(shape[2] - 1, shape[1] - 1, shape[0] - 1);
        Point3i min_voxel_index(p * resolution);

        return voxel_grad(p, min_voxel_index);
    }

    /// Very efficient normals (faceted appearance)
    Normal3f falcao(const Point3f& point) const {
        // FALCÃO , P., 2008. Implicit function to distance function.
        // URL: https://www.pouet.net/topic.php?which=5604&page=3#c233266.

        // FIXME: Something is numerically unstable ?!

        // Scale epsilon w.r.t inverse resolution
        auto shape = m_grid_texture.tensor().shape();
        Vector3f epsilon =
            0.1f * Vector3f(1.f / shape[2], 1.f / shape[1], 1.f / shape[0]);

        auto v = [&](const Point3f& p){
            Float out;
            m_grid_texture.eval(rescale_point(p), &out);
            return out;
        };

        Point3f p1(point.x() + epsilon.x(), point.y() - epsilon.y(), point.z() - epsilon.z());
        Point3f p2(point.x() - epsilon.x(), point.y() - epsilon.y(), point.z() + epsilon.z());
        Point3f p3(point.x() - epsilon.x(), point.y() + epsilon.y(), point.z() - epsilon.z());
        Point3f p4(point.x() + epsilon.x(), point.y() + epsilon.y(), point.z() + epsilon.z());

        Float v1 = v(p1);
        Float v2 = v(p2);
        Float v3 = v(p3);
        Float v4 = v(p4);

        Normal3f out = Normal3f(((v4 + v1) / 2.f) - ((v3 + v2) / 2.f),
                                ((v3 + v4) / 2.f) - ((v1 + v2) / 2.f),
                                ((v2 + v4) / 2.f) - ((v3 + v1) / 2.f));

        return dr::normalize(m_to_world.value().transform_affine(out));
    }

    static constexpr uint32_t optix_geometry_flags[1] = {
        OPTIX_GEOMETRY_FLAG_NONE
    };

    enum NormalMethod {
        Analytic,
        Smooth,
        Falcao,
    };

    enum Interpolation {
        Linear,
        Cubic,
    };

#if defined(MI_ENABLE_CUDA)
    void* m_optix_bboxes = nullptr;
    void* m_optix_voxel_indices = nullptr;
#endif
    // TODO: Store inverse shape using `rcp`
    Texture3f m_grid_texture;
    bool m_watertight;
    size_t m_filled_voxel_count = 0;
    NormalMethod m_normal_method;
    Interpolation m_interpolation;
};

MI_IMPLEMENT_CLASS_VARIANT(SDFGrid, SDF)
MI_EXPORT_PLUGIN(SDFGrid, "SDFGrid intersection primitive");
NAMESPACE_END(mitsuba)
