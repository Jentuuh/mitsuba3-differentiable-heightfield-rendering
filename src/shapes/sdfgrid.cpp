#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>
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

Documentation notes:
 * Grid position [0, 0, 0] x [1, 1, 1]
 * Reminder that tensors use [Z, Y, X, C] indexing
 * Does not emit UVs for texturing
 * Cannot be used for area emitters

 Temorary issues:
     * No props to initialize a grid
     * Rotations not allowed in `to_world`
     * Embree does not work
 */

template <typename Float, typename Spectrum>
class SDFGrid final : public Shape<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Shape, m_to_world, m_to_object, m_is_instance, initialize,
                   mark_dirty, get_children_string, parameters_grad_enabled)
    MI_IMPORT_TYPES()

    using typename Base::ScalarSize;

    SDFGrid(const Properties &props) : Base(props) {
        // TODO: Allow intitial grid in props
        float grid_data[8] = { -0.1f, 0.1f, 0.1f, 0.1f, 0.9f, 0.9f, 0.9f, 0.9f };
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
#endif
    }

    void update() {
        // TODO: Check for rotation - rotations are not allowed!
        m_to_object = m_to_world.value().inverse();
        mark_dirty();
   }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_parameter("to_world", *m_to_world.ptr(), ParamFlags::Differentiable | ParamFlags::Discontinuous);
        callback->put_parameter("grid", m_grid_texture.tensor(), ParamFlags::Differentiable | ParamFlags::Discontinuous);
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

    Normal3f normal(const Point3f& point) const {
        // FALCÃO , P., 2008. Implicit function to distance function.
        // URL: https://www.pouet.net/topic.php?which=5604&page=3#c233266.

        Float epsilon = dr::Epsilon<Float> * 100000;
        auto shape = m_grid_texture.tensor().shape();

        // TODO: make this more efficient
        Vector3f inv_shape(1.f / shape[2], 1.f / shape[1], 1.f / shape[0]);

        auto v = [&](const Point3f& p){
            using Array1f = dr::Array<Float, 1>;
            Array1f out;
            Point3f offset_p = p * (1 - inv_shape) + (inv_shape / 2.f);
            m_grid_texture.eval(offset_p, out.data());
            return out[0];
        };

        Point3f p1(point.x() + epsilon, point.y() - epsilon, point.z() - epsilon);
        Point3f p2(point.x() - epsilon, point.y() - epsilon, point.z() + epsilon);
        Point3f p3(point.x() - epsilon, point.y() + epsilon, point.z() - epsilon);
        Point3f p4(point.x() + epsilon, point.y() + epsilon, point.z() + epsilon);

        Float v1 = v(p1);
        Float v2 = v(p2);
        Float v3 = v(p3);
        Float v4 = v(p4);

        Normal3f out =
            dr::normalize(Vector3f(((v4 + v1) / 2.f) - ((v3 + v2) / 2.f),
                                   ((v3 + v4) / 2.f) - ((v1 + v2) / 2.f),
                                   ((v2 + v4) / 2.f) - ((v3 + v1) / 2.f)));

        return out;
    }

    SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                     const PreliminaryIntersection3f &pi,
                                                     uint32_t ray_flags,
                                                     uint32_t recursion_depth,
                                                     Mask active) const override {
        // TODO: support for RayFlags::DetachShape
        MI_MASK_ARGUMENT(active);
        constexpr bool IsDiff = dr::is_diff_v<Float>;

        // Early exit when tracing isn't necessary
        if (!m_is_instance && recursion_depth > 0)
            return dr::zeros<SurfaceInteraction3f>();

        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();

        bool follow_shape = has_flag(ray_flags, RayFlags::FollowShape);

        Transform4f to_world = m_to_world.value();
        Transform4f to_object = m_to_object.value();

        if constexpr (IsDiff) {
            if (follow_shape) {
                /* FollowShape glues the interaction point with the shape.
                   Therefore, to also account for a possible differential motion
                   of the shape, we first compute a detached intersection point
                   in local space and transform it back in world space to get a
                   point rigidly attached to the shape's motion, including
                   translation, scaling and rotation. */
                Point3f local = to_object.transform_affine(ray(pi.t));
                local = dr::detach(local);

                /* With FollowShape the local position should always be static as
                   the intersection point follows any motion of the sphere. */
                auto shape = m_grid_texture.tensor().shape();
                Vector3f inv_shape(1.f / shape[2], 1.f / shape[1], 1.f / shape[0]);
                using Array1f = dr::Array<Float, 1>;
                Array1f out;
                Point3f offset_p = local * (1 - inv_shape) + (inv_shape / 2.f);

                m_grid_texture.eval(offset_p, out.data());
                Float sdf_value = out[0];
                si.t = dr::replace_grad(pi.t, sdf_value); // TODO: needs projection along ray direction ?

                // Use local ray to capture gradients of `to_world`
                Ray3f ray_local = to_object.transform_affine(ray);
                ray_local = dr::detach(ray_local);
                si.p = to_world.transform_affine(ray_local(si.t));
            } else {
                // TODO: Need ray_intersect_preliminary_impl
                // it doesn't work otherwise
                si.t = pi.t;
                si.p = ray(si.t);
            }
        } else {
            // TODO: Can/need a reprojection?
            si.t = pi.t;
            si.p = ray(si.t);
        }

        si.t = dr::select(active, si.t, dr::Infinity<Float>);

        // TODO: check if we're not "double-counting" `to_world`'s gradient
        si.n = normal(m_to_object.value().transform_affine(si.p));
        si.sh_frame.n = si.n;

        si.uv = Point2f(0.f, 0.f);
        si.dp_du = Vector3f(0.f);
        si.dp_dv = Vector3f(0.f);
        si.dn_du = si.dn_dv = dr::zeros<Vector3f>();

        si.shape    = this;
        si.instance = nullptr;

        // TODO: boundary_test

        return si;
    }

    bool parameters_grad_enabled() const override {
        return dr::grad_enabled(m_to_world);
    }

#if defined(MI_ENABLE_CUDA)
    using Base::m_optix_data_ptr;

    void* build_bboxes() {
        // TODO: this can computed on-the-fly in the intersection shader, need to store the voxel index instead
        auto shape  = m_grid_texture.tensor().shape();
        float shape_rcp[3] = { 1.f / (shape[0] - 1), 1.f / (shape[1] - 1), 1.f / (shape[2] - 1) };
        size_t voxel_count = (shape[0] - 1) * (shape[1] - 1) * (shape[2] - 1);

        size_t voxel_index = 0;
        optix::BoundingBox3f* data = new optix::BoundingBox3f[voxel_count]();
        for (size_t z = 0; z < shape[0] - 1; ++z) {
            for (size_t y = 0; y < shape[1] - 1 ; ++y) {
                for (size_t x = 0; x < shape[2] - 1; ++x) {
                    ScalarBoundingBox3f bbox;
                    ScalarTransform4f to_world = m_to_world.scalar();

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

                    data[voxel_index++] = optix::BoundingBox3f(bbox);
                }
            }
        }

        void* data_ptr = jit_malloc(AllocType::Device, sizeof(optix::BoundingBox3f) * voxel_count);
        jit_memcpy(JitBackend::CUDA, data_ptr, data, sizeof(optix::BoundingBox3f) * voxel_count);

        return data_ptr;
    }

    void optix_prepare_geometry() override {
        if constexpr (dr::is_cuda_v<Float>) {
            // TODO: more efficient memory allocations
            if (m_optix_data_ptr)
                jit_free(m_optix_data_ptr);
            if (m_optix_bboxes)
                jit_free(m_optix_bboxes);

            size_t resolution[3] = { m_grid_texture.tensor().shape()[2],
                                     m_grid_texture.tensor().shape()[1],
                                     m_grid_texture.tensor().shape()[0] };

            m_optix_data_ptr = jit_malloc(AllocType::Device, sizeof(OptixSDFGridData));
            m_optix_bboxes = build_bboxes();

            OptixSDFGridData data = { (optix::BoundingBox3f *) m_optix_bboxes,
                                      resolution[0],
                                      resolution[1],
                                      resolution[2],
                                      m_grid_texture.tensor().array().data(),
                                      m_to_object.scalar() };

            jit_memcpy(JitBackend::CUDA, m_optix_data_ptr, &data, sizeof(OptixSDFGridData));
        }
    }

    void optix_build_input(OptixBuildInput &build_input) const override {
        //TODO: Only submit voxels with guaranteed surface in them
        auto shape = m_grid_texture.tensor().shape();
        size_t voxel_count = (shape[0] - 1) * (shape[1] - 1) * (shape[2] - 1);

        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

        build_input.customPrimitiveArray.aabbBuffers = &m_optix_bboxes;
        build_input.customPrimitiveArray.numPrimitives = voxel_count;
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
    static constexpr uint32_t optix_geometry_flags[1] = {
        OPTIX_GEOMETRY_FLAG_NONE
    };

#if defined(MI_ENABLE_CUDA)
    void* m_optix_bboxes;
#endif
    // TODO: Store inverse shape using `rcp`
    Texture3f m_grid_texture;
};

MI_IMPLEMENT_CLASS_VARIANT(SDFGrid, Shape)
MI_EXPORT_PLUGIN(SDFGrid, "SDFGrid intersection primitive");
NAMESPACE_END(mitsuba)