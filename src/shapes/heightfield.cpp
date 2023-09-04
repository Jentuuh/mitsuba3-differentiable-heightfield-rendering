#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>

#include <drjit/tensor.h>
#include <drjit/texture.h>

#if defined(MI_ENABLE_EMBREE)
#include <embree3/rtcore.h>
#endif

#if defined(MI_ENABLE_CUDA)
    #include "optix/heightfield.cuh"
#endif

NAMESPACE_BEGIN(mitsuba)

/**!

.. _shape-heightfield:

Heightfield (:monosp:`heightfield`)
----------------------------------------------------

The heightfield is defined as a rectangular surface aligned with the XY-plane, with displacements along the positive Z-axis. 
The surface's left bottom corner is at (-1.0,-1.0, 0.0), while the surface's right top corner is at (1.0, 1.0, 0.0). 
Optionally, the heightfield can be transformed by passing the `to_world` parameter. E.g.: We could scale it along the
X- and Y-axes.

.. pluginparameters::

 * - filename
   - |string|
   - Name of the heightfield texture that stores the heightfield. (Single color channel, .bmp file format)
   - |exposed|

 * - resolution
   - |int|
   - Resolution for default heightfield. In case `filename` wasn't provided, this resolution will be used to initialize a
    'flat' heightfield (all texels are initialized to 0.0). In case `filename` was provided, this parameter will be ignored. (Default = 128);

 * - max_height
   - |float|
   - Maximal height displacement of the heightfield. (Default = 1.0f);

 * - heightfield
   - |tensor|
   - Tensor array containing the heightfield data. (Shape: [bitmap.width(), bitmap.height(), 1])
   - |exposed|, |differentiable|, |discontinuous|

 * - to_world
   - |transform|
   - Specifies an optional linear object-to-world transformation. Note that non-uniform scales are
     not permitted! (Default: none, i.e. object space = world space)
   - |exposed|, |differentiable|, |discontinuous|

 * - per_vertex_n
   - |bool|
   - Maximal height displacement of the heightfield. (Default = true);

.. warning::

   - An heightfield shape does not emit UV coordinates for texturing.
   - Additionally, this shape can't be used as an area emitter


A simple example for instantiating a heightfield:

.. tabs::
    .. code-tab:: xml
        :name: heightfield

        <shape type="heightfield">
            <string name="filename" value="heightfield.png"/> 
        </shape>

    .. code-tab:: python
        {
        'type': 'heightfield',
        'material': {
            'type': 'diffuse'
        }
 */

template <typename Float, typename Spectrum>
class Heightfield final : public Shape<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Shape, m_to_world, m_to_object, m_is_instance, initialize,
                   mark_dirty, get_children_string, parameters_grad_enabled)
    MI_IMPORT_TYPES()

    // Heightfield texture is always stored in single precision
    using InputFloat     = dr::replace_scalar_t<Float, float>;
    using InputTexture2f = dr::Texture<InputFloat, 2>;
    using InputPoint1f   = Point<InputFloat, 1>;
    using InputTensorXf  = dr::Tensor<DynamicBuffer<InputFloat>>;
    using Index = typename CoreAliases::UInt32;

    // Triangulation types
    using InputPoint3f  = Point<float, 3>;
    using InputVector2f = Vector<float, 2>;
    using InputVector3f = Vector<float, 3>;
    using InputNormal3f = Normal<float, 3>;
    using FloatStorage = DynamicBuffer<dr::replace_scalar_t<Float, float>>;

    using typename Base::ScalarIndex;
    using typename Base::ScalarSize;

    Heightfield(const Properties &props) : Base(props) {        
        // Maximal height value of the heightfield (normalized height values will be scaled accordingly)
        m_max_height = props.get<ScalarFloat>("max_height", 1.0f);
        
        // ==================================
        // Load heightfield data from bitmap
        // ==================================
        if (props.has_property("filename")) {
            FileResolver *fs   = Thread::thread()->file_resolver();
            fs::path file_path = fs->resolve(props.string("filename"));
            if (!fs::exists(file_path))
                Log(Error, "\"%s\": file does not exist!", file_path);
            ref<Bitmap> heightfield_bitmap = new Bitmap(file_path);
            
            // Convert to float32 representation
            ref<Bitmap> normalized = heightfield_bitmap->convert(Bitmap::PixelFormat::Y, Struct::Type::Float32, false);

            m_res_x = normalized->width();
            m_res_y = normalized->height();
            m_vertex_count = m_res_x * m_res_y;

            // Tensor dimension should be texture dimension + 1 (--> for color channel)
            size_t shape[3] = {m_res_x, m_res_y, 1};
            
            m_heightfield_texture = InputTexture2f(InputTensorXf((float*)normalized->data(), 3, shape), true, false,
                dr::FilterMode::Linear, dr::WrapMode::Clamp);     

        } 
        // ================================================
        // Load flat heightfield with specified resolution
        // ================================================
        else {
            uint32_t resolution = props.get<int>("resolution", 128);

            m_res_x = resolution;
            m_res_y = resolution;
            m_vertex_count = m_res_x * m_res_y;

            std::vector<float> default_data = std::vector<float>(resolution * resolution);
 
            size_t default_shape[3] = { m_res_x, m_res_y, 1 };
            m_heightfield_texture = InputTexture2f(
                InputTensorXf(default_data.data(), 3, default_shape), true, false,
                dr::FilterMode::Linear, dr::WrapMode::Clamp);
        }

        // Per-vertex normal buffer
        m_has_vertex_normals = props.get<bool>("per_vertex_n", true);
        if (m_has_vertex_normals)
            m_vertex_normals = dr::zeros<FloatStorage>((m_res_x * m_res_y) * 3);

        update();
        initialize();
    }

    ~Heightfield() {
        jit_free(m_host_bboxes);
        jit_free(m_device_bboxes);
    }

    void  update() {
        m_to_object = m_to_world.value().inverse();

        if constexpr (!dr::is_cuda_v<Float>) {
            dr::eval(m_heightfield_texture.value()); // Make sure the heightfield data is evaluated
            m_host_grid_data = m_heightfield_texture.tensor().data();
        }

        jit_free(m_host_bboxes);
        jit_free(m_device_bboxes);
        std::tie(m_host_bboxes,
                 m_device_bboxes,
                 m_amount_primitives) = build_bboxes();

        // Update per-vertex normals
        if(m_has_vertex_normals)
            recompute_vertex_normals();

        mark_dirty();
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_parameter("to_world", *m_to_world.ptr(), ParamFlags::Differentiable | ParamFlags::Discontinuous);
        callback->put_parameter("heightfield", m_heightfield_texture.tensor(), ParamFlags::Differentiable | ParamFlags::Discontinuous);
        callback->put_parameter("max_height", *m_max_height.ptr(), +ParamFlags::NonDifferentiable);
        callback->put_parameter("per_vertex_normals", m_vertex_normals, +ParamFlags::NonDifferentiable);
        callback->put_parameter("res_x", m_res_x, +ParamFlags::NonDifferentiable);
        callback->put_parameter("res_y", m_res_y, +ParamFlags::NonDifferentiable);
    }

    void parameters_changed(const std::vector<std::string> &keys) override {
        if (keys.empty() || string::contains(keys, "to_world") || string::contains(keys, "heightfield") || string::contains(keys, "max_height")) {
            // Ensure previous ray-tracing operation are fully evaluated before
            // modifying the scalar values of the fields in this class
            if constexpr (dr::is_jit_v<Float>)
                dr::sync_thread();

            // Update the scalar value of the matrix
            m_to_world = m_to_world.value();

            // Update heightfield texture
            m_heightfield_texture.set_tensor(m_heightfield_texture.tensor());

            // Update m_max_height
            m_max_height = m_max_height.value();

            // Update vertex count and vertex normals storage in case the heightfield resolution changed
            bool mesh_attributes_changed = false;
            if(m_heightfield_texture.tensor().shape(0) * m_heightfield_texture.tensor().shape(1) != m_vertex_count)
            {
                Log(Debug, "parameters_changed(): Heightfield vertex count (resolution) changed, updating it.");
                mesh_attributes_changed = true;
                m_res_x = m_heightfield_texture.tensor().shape(0);
                m_res_y = m_heightfield_texture.tensor().shape(1);
                m_vertex_count = m_res_x * m_res_y;
            }
            if (m_has_vertex_normals && m_vertex_normals.size() != m_vertex_count * 3) {
                Log(Debug, "parameters_changed(): Heightfield vertex normal count changed, updating it.");
                mesh_attributes_changed = true;
                m_vertex_normals = dr::zeros<FloatStorage>(m_vertex_count * 3);
            }

            update();
        }

        Base::parameters_changed();
    }


    /* \brief Computes AABBs for all heightfield tiles (a heightfield tile 
     * implicitly contains a surface inside). Returns a pointer to the array 
     * of AABBs, a pointer to an array of cell indices of the former AABBs and the
     * amount of AABBs that were initialized.
     */
    std::tuple<void *, void *, size_t> build_bboxes() {
        // Bbox count is (m_res_x - 1) * (m_res_y - 1) because our tensor data represents heightfield values 
        // at tile corners. (4 input texture texels form 1 heightfield tile)
        size_t max_bbox_count =
            (m_res_x - 1) * (m_res_y - 1);

        float *grid = nullptr;
        
        if constexpr (dr::is_cuda_v<Float>) {
            grid = (float *) jit_malloc_migrate(
                m_heightfield_texture.tensor().array().data(), AllocType::Host, false);
            jit_sync_thread();
        } else {
            grid = m_heightfield_texture.tensor().array().data();
        }

        #if defined(MI_ENABLE_CUDA)
             using BoundingBoxType =
            typename std::conditional<dr::is_cuda_v<Float>,
                                      optix::BoundingBox3f,
                                      ScalarBoundingBox3f>::type;
        #else 
            using BoundingBoxType = ScalarBoundingBox3f;
        #endif

        BoundingBoxType *host_aabbs = (BoundingBoxType *) jit_malloc(
        AllocType::Host, sizeof(BoundingBoxType) * max_bbox_count);

        // Loop over heightfield tiles and build AABB for each
        size_t count = 0;
        ScalarPoint3f tri_1[3];
        ScalarPoint3f tri_2[3];
        Vector4u indices;
        for (size_t tile = 0; tile < max_bbox_count; tile++) {
            ScalarBoundingBox3f bbox;
            get_tile_vertices_scalar(tile, tri_1, tri_2, indices, grid);
            bbox.expand(tri_1[0]); // Left top vertex
            bbox.expand(tri_1[1]); // Left bottom vertex
            bbox.expand(tri_1[2]); // Right bottom vertex
            bbox.expand(tri_2[0]); // Left top vertex
            host_aabbs[count] = BoundingBoxType(bbox);
            count++;
        }

        // Upload to device (if applicable)
        void *device_aabbs = nullptr;
        if constexpr (dr::is_cuda_v<Float>) {
            device_aabbs =
                jit_malloc(AllocType::Device, sizeof(BoundingBoxType) * count);
            jit_memcpy_async(JitBackend::CUDA, device_aabbs, host_aabbs,
                        sizeof(BoundingBoxType) * count);

            jit_free(grid);
        }

        return { host_aabbs, device_aabbs, count };
    }

    ScalarSize primitive_count() const override { return m_amount_primitives; }


    /**
     * The heightfield contains a `m_max_height` member that forms a boundary on the maximum vertical displacement of heightfield texels.
     * This allows us to easily build a bounding box that encapsulates the entire heightfield.
     **/
    ScalarBoundingBox3f bbox() const override {
        ScalarBoundingBox3f bbox;
        ScalarTransform4f to_world = m_to_world.scalar();

        bbox.expand(to_world.transform_affine(ScalarPoint3f(-1.f, -1.f, 0.f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f(-1.f,  1.f, 0.f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f( 1.f, -1.f, 0.f)));
        bbox.expand(to_world.transform_affine(ScalarPoint3f( 1.f,  1.f, 0.f)));

        bbox.expand(to_world.transform_affine(ScalarPoint3f(-1.f, -1.f, m_max_height.scalar())));
        bbox.expand(to_world.transform_affine(ScalarPoint3f(-1.f,  1.f, m_max_height.scalar())));
        bbox.expand(to_world.transform_affine(ScalarPoint3f( 1.f, -1.f, m_max_height.scalar())));
        bbox.expand(to_world.transform_affine(ScalarPoint3f( 1.f,  1.f, m_max_height.scalar())));
        return bbox;
    }

    ScalarBoundingBox3f bbox(ScalarIndex index) const override {
        if constexpr (dr::is_cuda_v<Float>) {
            NotImplementedError("bbox(ScalarIndex index)");
        }

        return reinterpret_cast<ScalarBoundingBox3f*>(m_host_bboxes)[index];
    }

    Float surface_area() const override {
        // TODO
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
                                   ScalarIndex prim_index,
                                   dr::mask_t<FloatP> active) const {
        MI_MASK_ARGUMENT(active);
        using Point2fP = Point<FloatP, 2>;
        using Point3fP = Point<FloatP, 3>;

         // 4 vertices of candidate heightfield tile 
        Point3fP t1[3];
        Point3fP t2[3];
        Vector4u indices;
        get_tile_vertices_scalar_packet(prim_index, t1, t2, indices);

        FloatP t;
        Point2fP uv;
        dr::mask_t<FloatP> active_t;
        
        // Give closest intersection between two triangles (if any)              
        std::tie(t, uv , active_t) = moeller_trumbore_two_triangles(ray_, t1, t2);
         
        return { dr::select(active_t, t, dr::Infinity<FloatP>),
            Point2fP(uv.x(), uv.y()), ((uint32_t) -1), prim_index };
    }


    template <typename FloatP, typename Ray3fP>
    dr::mask_t<FloatP> ray_test_impl(const Ray3fP &ray_,
                                     ScalarIndex prim_index,
                                     dr::mask_t<FloatP> active) const {
        MI_MASK_ARGUMENT(active);
        using Point2fP = Point<FloatP, 2>;
        using Point3fP = Point<FloatP, 3>;   

        // 4 vertices of candidate heightfield tile 
        Point3fP t1[3];
        Point3fP t2[3];
        Vector4u indices;
        get_tile_vertices_scalar_packet(prim_index, t1, t2, indices);

        FloatP t;
        Point2fP uv;
        dr::mask_t<FloatP> active_t;
        
        // Give closest intersection between two triangles (if any)              
        std::tie(t, uv , active_t) = moeller_trumbore_two_triangles(ray_, t1, t2);
         
        return active_t;
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

        bool detach_shape = has_flag(ray_flags, RayFlags::DetachShape);
        bool follow_shape = has_flag(ray_flags, RayFlags::FollowShape);

        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();

        Float t = pi.t;
        si.p = ray(pi.t);
        Point2f prim_uv = pi.prim_uv;

        // Compute which triangle we intersected with           TODO: We currently decided to do this on the fly, might be better to pass the index via prim_index, or add another return value to the tuple that's returned from `ray_intersect_preliminary_impl`
        //  -------
        //  | \neg|
        //  |  \  |    Diagonal plane equation outcome mapping
        //  |pos\ |
        //  -------
        Point3f t1[3];
        Point3f t2[3];
        Vector4u indices;
        get_tile_vertices_vectorized(pi.prim_index, t1, t2, indices);
        
        Vector3f diagonal = t1[2] - t1[0];
        Vector3f diag_normal = dr::cross(diagonal, Vector3f{0.0f, 0.0f, -1.0f});
        Float D = -(dr::dot(diag_normal, t1[0]));
        Float above_or_below_diagonal_plane = dr::dot(diag_normal, si.p) + D;

        Point3f hit_tri[3];
        hit_tri[0] = dr::select(above_or_below_diagonal_plane > 0 && active, t1[0], t2[0]);
        hit_tri[1] = dr::select(above_or_below_diagonal_plane > 0 && active, t1[1], t2[1]);
        hit_tri[2] = dr::select(above_or_below_diagonal_plane > 0 && active, t1[2], t2[2]);

        if constexpr (IsDiff) {
            if (follow_shape && detach_shape)
                Throw("Invalid combination of RayFlags: DetachShape | FollowShape");

            if (detach_shape) {
                hit_tri[0] = dr::detach<true>(hit_tri[0]);
                hit_tri[1] = dr::detach<true>(hit_tri[1]);
                hit_tri[2] = dr::detach<true>(hit_tri[2]);
            }

            if(dr::grad_enabled(hit_tri[0], hit_tri[1], hit_tri[2], ray.o, ray.d /* <- any enabled? */) && !follow_shape)
            {
                auto [t_d, prim_uv_d, hit] = moeller_trumbore_two_triangles(ray, t1, t2);

                prim_uv = dr::replace_grad(prim_uv, prim_uv_d);
                t = dr::replace_grad(t, t_d);
            }
        }
        
        // Barycentric coords
        Float b1 = prim_uv.x(),
              b2 = prim_uv.y(),
              b0 = 1.f - b1 - b2;

        Vector3f dp0 = hit_tri[1] - hit_tri[0],
                 dp1 = hit_tri[2] - hit_tri[0];

        // Re-interpolate intersection using barycentric coordinates
        si.p = dr::fmadd(hit_tri[0], b0, dr::fmadd(hit_tri[1], b1, hit_tri[2] * b2));

        // Potentially recompute the distance traveled to the surface interaction hit point
        if (IsDiff && follow_shape)
            t = dr::sqrt(dr::squared_norm(si.p - ray.o) / dr::squared_norm(ray.d));

        si.t = dr::select(active, t, dr::Infinity<Float>);

        // Face normal 
        si.n  = dr::normalize(dr::cross(dp0, dp1));

        // Shading normals
        Normal3f n0, n1, n2;
        if(m_has_vertex_normals && 
            likely(has_flag(ray_flags, RayFlags::ShadingFrame) ||
                   has_flag(ray_flags, RayFlags::dNSdUV) ||
                   has_flag(ray_flags, RayFlags::BoundaryTest))) {
            n0 = dr::select(above_or_below_diagonal_plane > 0, vertex_normal(indices[0]), vertex_normal(indices[3]));
            n1 = dr::select(above_or_below_diagonal_plane > 0, vertex_normal(indices[1]), vertex_normal(indices[0]));
            n2 = dr::select(above_or_below_diagonal_plane > 0, vertex_normal(indices[2]), vertex_normal(indices[2]));
        }

        if (m_has_vertex_normals &&
            likely(has_flag(ray_flags, RayFlags::ShadingFrame) ||
                   has_flag(ray_flags, RayFlags::dNSdUV))) {

            if (IsDiff && detach_shape) {
                n0 = dr::detach<true>(n0);
                n1 = dr::detach<true>(n1);
                n2 = dr::detach<true>(n2);
            }

            Normal3f n = dr::fmadd(n2, b2, dr::fmadd(n1, b1, n0 * b0));
            Float il = dr::rsqrt(dr::squared_norm(n));
            n *= il;

            si.sh_frame.n = dr::normalize(n);

            if (has_flag(ray_flags, RayFlags::dNSdUV)) {
                /* Now compute the derivative of "normalize(u*n1 + v*n2 + (1-u-v)*n0)"
                with respect to [u, v] in the local triangle parameterization.

                Since d/du [f(u)/|f(u)|] = [d/du f(u)]/|f(u)|
                    - f(u)/|f(u)|^3 <f(u), d/du f(u)>, this results in
                */
                si.dn_du = (n1 - n0) * il;
                si.dn_dv = (n2 - n0) * il;

                si.dn_du = dr::fnmadd(n, dr::dot(n, si.dn_du), si.dn_du);
                si.dn_dv = dr::fnmadd(n, dr::dot(n, si.dn_dv), si.dn_dv);
            } else {
                si.dn_du = si.dn_dv = dr::zeros<Vector3f>();
            }     
        } else {
            si.sh_frame.n = si.n;         
        }

        // Global UV's (heightmap space)
        si.uv = Point2f((si.p.x() + 1.0f) / 2.0f, (si.p.y() + 1.0f) / 2.0f);

        // Positional partial derivative
        si.dp_du      = dr::zeros<Vector3f>();
        si.dp_dv      = dr::zeros<Vector3f>();
        
        si.shape    = this;
        si.instance = nullptr;

        if (unlikely(has_flag(ray_flags, RayFlags::BoundaryTest))) {
            Vector3f rel = si.p - hit_tri[0];

            /* Solve a least squares problem to determine
            the UV coordinates within the current triangle */
            Float bb1 = dr::dot(dp0, rel),
                  bb2 = dr::dot(dp1, rel),
                  a11 = dr::dot(dp0, dp0),
                  a12 = dr::dot(dp0, dp1),
                  a22 = dr::dot(dp1, dp1),
                  inv_det = dr::rcp(a11 * a22 - a12 * a12);

            Float u = dr::fmsub (a22, bb1, a12 * bb2) * inv_det,
                  v = dr::fnmadd(a12, bb1, a11 * bb2) * inv_det,
                  w = 1.f - u - v;

            /* If we are using flat shading, just fall back to a signed distance
            field of the hit triangle. */
            if (!m_has_vertex_normals) {
                // 2D Triangle SDF from Inigo Quilez
                // https://www.iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm

                // Equilateral triangle
                Point2f tp0 = Point2f(0, 0),
                        tp1 = Point2f(1, 0),
                        tp2 = Point2f(0.5f, 0.5f * dr::sqrt(3.f));

                Point2f p = tp0 * w + tp1 * u + tp2 * v;

                Vector2f e0 = tp1 - tp0,
                         e1 = tp2 - tp1,
                         e2 = tp0 - tp2,
                         v0 = p - tp0,
                         v1 = p - tp1,
                         v2 = p - tp2;
                Vector2f pq0 = v0 - e0 * dr::clamp(dr::dot(v0, e0) / dr::dot(e0, e0), 0, 1),
                         pq1 = v1 - e1 * dr::clamp(dr::dot(v1, e1) / dr::dot(e1, e1), 0, 1),
                         pq2 = v2 - e2 * dr::clamp(dr::dot(v2, e2) / dr::dot(e2, e2), 0, 1);
                Float s = dr::sign(e0.x() * e2.y() - e0.y() * e2.x());
                Vector2f d = dr::minimum(dr::minimum(Vector2f(dr::dot(pq0, pq0), s * (v0.x() * e0.y() - v0.y() * e0.x())),
                                                     Vector2f(dr::dot(pq1, pq1), s * (v1.x() * e1.y() - v1.y() * e1.x()))),
                                                     Vector2f(dr::dot(pq2, pq2), s * (v2.x() * e2.y() - v2.y() * e2.x())));
                Float dist = dr::sqrt(d.x());
                // Scale s.t. farthest point / barycenter is one
                dist /= dr::sqrt(3.f) / 6.f;
                si.boundary_test = dist;
            } else {
                Normal3f normal = dr::fmadd(n0, w, dr::fmadd(n1, u, n2 * v));

                // Dot product between surface normal and the ray direction is 0 at silhouette points
                Float dp = dr::dot(normal, -ray.d);

                // Add non-linearity by squaring the returned value
                si.boundary_test = dr::sqr(dp);
            }
        }
        
        return si;
    }
    
    template <typename FloatP, typename Ray3fP>
    MI_INLINE std::tuple<FloatP,  Point<FloatP, 2>, dr::mask_t<FloatP>> moeller_trumbore_two_triangles(const Ray3fP &ray, Point<FloatP, 3> tri_1[3], Point<FloatP, 3> tri_2[3],
                                                                                               dr::mask_t<FloatP> active = true) const 
    {
        using Vector3fP = Vector<FloatP, 3>;

        dr::mask_t<FloatP> active1 = active;
        dr::mask_t<FloatP> active2 = active;

        Vector3fP e1t1 = tri_1[1] - tri_1[0], e1t2 = tri_2[1] - tri_2[0], e2t1 = tri_1[2] - tri_1[0], e2t2 = tri_2[2] - tri_2[0];

        Vector3fP pvec_t1 = dr::cross(ray.d, e2t1);
        Vector3fP pvec_t2 = dr::cross(ray.d, e2t2);
        FloatP inv_det_t1 = dr::rcp(dr::dot(e1t1, pvec_t1));
        FloatP inv_det_t2 = dr::rcp(dr::dot(e1t2, pvec_t2));

        Vector3fP tvec_t1 = ray.o - tri_1[0];
        Vector3fP tvec_t2 = ray.o - tri_2[0];

        FloatP u1 = dr::dot(tvec_t1, pvec_t1) * inv_det_t1;  // ok
        FloatP u2 = dr::dot(tvec_t2, pvec_t2) * inv_det_t2;  
        active1 &= u1 >= 0.f && u1 <= 1.f;
        active2 &= u2 >= 0.f && u2 <= 1.f;

        Vector3fP qvec1 = dr::cross(tvec_t1, e1t1);
        Vector3fP qvec2 = dr::cross(tvec_t2, e1t2);
        FloatP v1 = dr::dot(ray.d, qvec1) * inv_det_t1;
        FloatP v2 = dr::dot(ray.d, qvec2) * inv_det_t2;
        active1 &= v1 >= 0.f && u1 + v1 <= 1.f;
        active2 &= v2 >= 0.f && u2 + v2 <= 1.f;

        FloatP t1 = dr::dot(e2t1, qvec1) * inv_det_t1;
        FloatP t2 = dr::dot(e2t2, qvec2) * inv_det_t2;
        active1 &= t1 >= 0.f && t1 <= ray.maxt;
        active2 &= t2 >= 0.f && t2 <= ray.maxt;

        // We take t1 if both t1 and t2 are valid and t1 is closer, OR if t1 is valid and t2 is not (the closest valid intersection)
        dr::mask_t<FloatP> hit_triangle_mask = (t1 < t2 && (active1 && active2)) || (active1 && !active2);

        // Select the closest triangle (if any intersection was found, this is covered by `closest_mask` below)
        FloatP closest_t = dr::select(hit_triangle_mask, t1, t2);
        FloatP closest_u = dr::select(hit_triangle_mask, u1, u2);
        FloatP closest_v = dr::select(hit_triangle_mask, v1, v2);

        // If it found an intersection, the resulting mask should always evaluate to true, t_value + uv-value 
        // of the closest intersection is already handled above. Therefore just `active1 OR active2`.
        dr::mask_t<FloatP>closest_mask = active1 || active2;
        return { closest_t, { closest_u, closest_v }, closest_mask };
    }


    // TODO: merge scalar and vectorized versions in a clean and non-redundant manner!
    /**
     * Get world-space vertices of a heightfield tile (scalar mode)
     * Tile indexing example (indexing starts on the left bottom of the grid,
     * goes lef):
     * 
     * x == -1        x == 1                Triangle 1: Left top, left bottom, right bottom (0,1,2)
     *                                      0-------
     * ^                ^                   | \    |
     * |                |                   |  \   |
     * |                |                   |   \  |    
     * ------------------ --> y == 1        1------2 
     * |    |     |     |                   
     * | 6  | 7   | 8   |                   Triangle 2: Right top , left top, right bottom (3,0,2)           
     * |----------------|                   0------3
     * |    |     |     |                   | \    |
     * | 3  | 4   | 5   |                   |  \   |
     * |----------------|                   |   \  |
     * |    |     |     |                   -------2
     * | 0  | 1   | 2   |
     * ------------------ --> y == -1
     * */
    template <typename FloatP>
    MI_INLINE void get_tile_vertices_scalar_packet(ScalarIndex prim_index, Point<FloatP, 3>* t1, Point<FloatP, 3>* t2, Vector4u& idx) const
    {
        using Point2fP = Point<FloatP, 2>;
        using Point3fP = Point<FloatP, 3>;

        float cell_size[2] = { 2.0f / (m_res_x - 1), 2.0f / (m_res_y - 1)};

        uint32_t amount_rows = m_res_x - 1;
        uint32_t amount_bboxes_per_row = m_res_y - 1;
        uint32_t values_per_row = m_res_x;
        uint32_t row_nr = dr::floor((float)prim_index / (float) amount_bboxes_per_row); // floor(prim_index / amount_bboxes_per_row)
        uint32_t row_offset = prim_index % (amount_bboxes_per_row); // prim_index % amount_bboxes_per_row

        // `(amount_rows - row_nr) * values_per_row` gives us the offset to get to the current row, we add the 
        // row offset to this to get the absolute offset to obtain the corresponding texel of the
        // current AABB (+ 1 in both dimensions to get right and top texels)  
        uint32_t left_bottom_index = (amount_rows - row_nr) * values_per_row + row_offset;
        uint32_t right_bottom_index = (amount_rows - row_nr) * values_per_row + row_offset + 1;
        uint32_t left_top_index = (amount_rows - (row_nr + 1)) * values_per_row + row_offset;
        uint32_t right_top_index = (amount_rows - (row_nr + 1)) * values_per_row + row_offset + 1;

        // Compute the fractional bounds of the tile we're testing the intersection for
        Point2fP local_min_target_bounds = Point2fP(-1.0f + (float)row_offset * cell_size[0], -1.0f + (float)row_nr * cell_size[1]);
        Point2fP local_max_target_bounds = Point2fP(-1.0f + (float)(row_offset + 1) * cell_size[0], -1.0f + (float)(row_nr + 1) * cell_size[1]);

        // 0 --> 1 --> 2
        t1[0] = m_to_world.scalar().transform_affine(Point3fP{local_min_target_bounds.x(), local_max_target_bounds.y(), m_host_grid_data[left_top_index] * m_max_height.scalar()});
        t1[1] = m_to_world.scalar().transform_affine(Point3fP{local_min_target_bounds.x(), local_min_target_bounds.y(), m_host_grid_data[left_bottom_index] * m_max_height.scalar()});
        t1[2] = m_to_world.scalar().transform_affine(Point3fP{local_max_target_bounds.x(), local_min_target_bounds.y(), m_host_grid_data[right_bottom_index] * m_max_height.scalar()});

        // 3 --> 0 --> 2
        t2[0] = m_to_world.scalar().transform_affine(Point3fP{local_max_target_bounds.x(), local_max_target_bounds.y(), m_host_grid_data[right_top_index] * m_max_height.scalar()});      
        t2[1] = m_to_world.scalar().transform_affine(Point3fP{local_min_target_bounds.x(), local_max_target_bounds.y(), m_host_grid_data[left_top_index] * m_max_height.scalar()});
        t2[2] = m_to_world.scalar().transform_affine(Point3fP{local_max_target_bounds.x(), local_min_target_bounds.y(), m_host_grid_data[right_bottom_index] * m_max_height.scalar()});
        
        idx[0] = left_top_index;
        idx[1] = left_bottom_index;
        idx[2] = right_bottom_index;
        idx[3] = right_top_index;
    }

    // We have to pass `float* grid` to this version, because both the LLVM/scalar and CUDA versions make use of it in `build_bboxes()`.
    // The `m_host_grid_data` weak pointer is only set in LLVM/scalar modes. 
    MI_INLINE void get_tile_vertices_scalar(ScalarIndex prim_index, ScalarPoint3f* t1, ScalarPoint3f* t2, Vector4u& idx, float* grid) const
    {
        float cell_size[2] = { 2.0f / (m_res_x - 1), 2.0f / (m_res_y - 1)};
        ScalarTransform4f to_world = m_to_world.scalar();

        uint32_t amount_rows = m_res_x - 1;
        uint32_t amount_bboxes_per_row = m_res_y - 1;
        uint32_t values_per_row = m_res_x;
        uint32_t row_nr = dr::floor((float)prim_index / (float) amount_bboxes_per_row);
        uint32_t row_offset = prim_index % (amount_bboxes_per_row);

        // `(amount_rows - row_nr) * values_per_row` gives us the offset to get to the current row, we add the 
        // row offset to this to get the absolute offset to obtain the corresponding texel of the
        // current AABB (+ 1 in both dimensions to get right and top texels)  
        uint32_t left_bottom_index = (amount_rows - row_nr) * values_per_row + row_offset;
        uint32_t right_bottom_index = (amount_rows - row_nr) * values_per_row + row_offset + 1;
        uint32_t left_top_index = (amount_rows - (row_nr + 1)) * values_per_row + row_offset;
        uint32_t right_top_index = (amount_rows - (row_nr + 1)) * values_per_row + row_offset + 1;

        // Compute the fractional bounds of the tile we're testing the intersection for
        ScalarPoint2f local_min_target_bounds = ScalarPoint2f(-1.0f + (float)row_offset * cell_size[0], -1.0f + (float)row_nr * cell_size[1]);
        ScalarPoint2f local_max_target_bounds = ScalarPoint2f(-1.0f + (float)(row_offset + 1) * cell_size[0], -1.0f + (float)(row_nr + 1) * cell_size[1]);

        // 0 --> 1 --> 2
        t1[0] = to_world.transform_affine(ScalarPoint3f{local_min_target_bounds.x(), local_max_target_bounds.y(), grid[left_top_index] * m_max_height.scalar()});
        t1[1] = to_world.transform_affine(ScalarPoint3f{local_min_target_bounds.x(), local_min_target_bounds.y(), grid[left_bottom_index] * m_max_height.scalar()});
        t1[2] = to_world.transform_affine(ScalarPoint3f{local_max_target_bounds.x(), local_min_target_bounds.y(), grid[right_bottom_index] * m_max_height.scalar()});

        // 3 --> 0 --> 2
        t2[0] = to_world.transform_affine(ScalarPoint3f{local_max_target_bounds.x(), local_max_target_bounds.y(), grid[right_top_index] * m_max_height.scalar()});      
        t2[1] = to_world.transform_affine(ScalarPoint3f{local_min_target_bounds.x(), local_max_target_bounds.y(), grid[left_top_index] * m_max_height.scalar()});
        t2[2] = to_world.transform_affine(ScalarPoint3f{local_max_target_bounds.x(), local_min_target_bounds.y(), grid[right_bottom_index] * m_max_height.scalar()});
        
        idx[0] = left_top_index;
        idx[1] = left_bottom_index;
        idx[2] = right_bottom_index;
        idx[3] = right_top_index;
    }

    /**
     * Get world-space vertices of a heightfield tile (vectorized mode)
     * */
       MI_INLINE void get_tile_vertices_vectorized(Index prim_index, Point3f* t1, Point3f* t2, Vector4u& idx) const
    {
        float cell_size[2] = { 2.0f / (m_res_x - 1), 2.0f / (m_res_y - 1)};

        UInt32 amount_rows = m_res_x - 1;
        UInt32 amount_bboxes_per_row = m_res_y - 1;
        UInt32 values_per_row = m_res_x;
        UInt32 row_nr = dr::floor((Float)prim_index / (Float)amount_bboxes_per_row);
        UInt32 row_offset = prim_index % (amount_bboxes_per_row); 

        // `(amount_rows - row_nr) * values_per_row` gives us the offset to get to the current row, we add the 
        // row offset to this to get the absolute offset to obtain the corresponding texel of the
        // current AABB (+ 1 in both dimensions to get right and top texels)  
        UInt32 left_bottom_index = (amount_rows - row_nr) * values_per_row + row_offset;
        UInt32 right_bottom_index = (amount_rows - row_nr) * values_per_row + row_offset + 1;
        UInt32 left_top_index = (amount_rows - (row_nr + 1)) * values_per_row + row_offset;
        UInt32 right_top_index = (amount_rows - (row_nr + 1)) * values_per_row + row_offset + 1;

        // Compute the fractional bounds of the tile we're testing the intersection for
        Point2f local_min_target_bounds = Point2f(-1.0f + row_offset * cell_size[0], -1.0f + row_nr * cell_size[1]);
        Point2f local_max_target_bounds = Point2f(-1.0f + (row_offset + 1) * cell_size[0], -1.0f + (row_nr + 1) * cell_size[1]);
        
        // 0 --> 1 --> 2
        t1[0] = m_to_world.value().transform_affine(Point3f{local_min_target_bounds.x(), local_max_target_bounds.y(), dr::gather<Float>(m_heightfield_texture.value(), left_top_index) * m_max_height.scalar()});
        t1[1] = m_to_world.value().transform_affine(Point3f{local_min_target_bounds.x(), local_min_target_bounds.y(), dr::gather<Float>(m_heightfield_texture.value(), left_bottom_index) * m_max_height.scalar()});
        t1[2] = m_to_world.value().transform_affine(Point3f{local_max_target_bounds.x(), local_min_target_bounds.y(), dr::gather<Float>(m_heightfield_texture.value(), right_bottom_index) * m_max_height.scalar()});

        // 3 --> 0 --> 2
        t2[0] = m_to_world.value().transform_affine(Point3f{local_max_target_bounds.x(), local_max_target_bounds.y(), dr::gather<Float>(m_heightfield_texture.value(), right_top_index) * m_max_height.scalar()});
        t2[1] = m_to_world.value().transform_affine(Point3f{local_min_target_bounds.x(), local_max_target_bounds.y(), dr::gather<Float>(m_heightfield_texture.value(), left_top_index) * m_max_height.scalar()});
        t2[2] = m_to_world.value().transform_affine(Point3f{local_max_target_bounds.x(), local_min_target_bounds.y(), dr::gather<Float>(m_heightfield_texture.value(), right_bottom_index) * m_max_height.scalar()});
        
        idx[0] = left_top_index;
        idx[1] = left_bottom_index;
        idx[2] = right_bottom_index;
        idx[3] = right_top_index;
    }


    void recompute_vertex_normals() {
        
        /* Weighting scheme based on "Computing Vertex Normals from Polygonal Facets"
       by Grit Thuermer and Charles A. Wuethrich, JGT 1998, Vol 3 */
       uint32_t vertex_count = m_res_x * m_res_y;

        if constexpr (!dr::is_dynamic_v<Float>) {
        size_t invalid_counter = 0;
        std::vector<InputNormal3f> normals(vertex_count, dr::zeros<InputNormal3f>());

            for(ScalarSize tile = 0; tile < (m_res_x - 1) * (m_res_y - 1); tile++){
                InputPoint3f triangle1[3]; 
                InputPoint3f triangle2[3];
                Vector4u indices;
                get_tile_vertices_scalar_packet(tile, triangle1, triangle2, indices);

                Vector3u indices_t1 = Vector3u{indices[0], indices[1], indices[2]}; // Store indices for triangle 2 in separate vector to allow easy indexing in loop
                Vector3u indices_t2 = Vector3u{indices[3], indices[0], indices[2]}; // Store indices for triangle 2 in separate vector to allow easy indexing in loop
                
                // Apply angle weighting scheme for both triangles in tile
                update_vertex_normals_scalar(triangle1, normals, indices_t1); 
                update_vertex_normals_scalar(triangle2, normals, indices_t2);
            }

            // Normalize all normals
            for (ScalarSize i = 0; i < vertex_count; i++) {
                InputNormal3f n = normals[i];
                InputFloat length = dr::norm(n);

                if (likely(length != 0.f)) {
                    n /= length;
                } else {
                    n = InputNormal3f(1, 0, 0); // Choose some bogus value
                    invalid_counter++;
                }

                dr::store(m_vertex_normals.data() + 3 * i, n);
            }

            if (invalid_counter > 0)
                Log(Warn, "Heightfield: computed vertex normals (%i invalid vertices!)",
                    invalid_counter);
        } else {
            // The following is JITed into two separate kernel launches

            // --------------------- Kernel 1 starts here ---------------------
            UInt32 tile_index = dr::arange<UInt32>((m_res_x - 1) * (m_res_y - 1));

            Point3f triangle1[3];
            Point3f triangle2[3];
            Vector4u indices;
            get_tile_vertices_vectorized(tile_index, triangle1, triangle2, indices);
            Vector3u indices_t1 = Vector3u{indices[0], indices[1], indices[2]}; // Store indices for triangle 2 in separate vector to allow easy indexing in loop
            Vector3u indices_t2 = Vector3u{indices[3], indices[0], indices[2]}; // Store indices for triangle 2 in separate vector to allow easy indexing in loop
    
            Vector3f normals = dr::zeros<Vector3f>(vertex_count);

            update_vertex_normals_vectorized(triangle1, normals, indices_t1);
            update_vertex_normals_vectorized(triangle2, normals, indices_t2);

            // --------------------- Kernel 2 starts here ---------------------
            normals = dr::normalize(normals);

            // Disconnect the vertex normal buffer from any pre-existing AD
            // graph. Otherwise an AD graph might be unnecessarily retained
            // here, despite the following lines re-initializing the normals.
            dr::disable_grad(m_vertex_normals);

            UInt32 ni = dr::arange<UInt32>(vertex_count) * 3;
            for (size_t i = 0; i < 3; ++i)
                dr::scatter(m_vertex_normals,
                            dr::float32_array_t<Float>(normals[i]), ni + i);

            dr::eval(m_vertex_normals);
        }
    }   


    // TODO: Merge scalar and vectorized versions into 1 helper function? 
    void update_vertex_normals_scalar(InputPoint3f* triangle, std::vector<InputNormal3f>& vertex_normals, Vector3u& indices)
    {
        if constexpr (!dr::is_dynamic_v<Float>) {
            InputVector3f   side_0 = triangle[1] - triangle[0],
                            side_1 = triangle[2] - triangle[0];
            InputNormal3f face_normal = dr::cross(side_0, side_1);
            
            InputFloat length_sqr = dr::squared_norm(face_normal);
            if (likely(length_sqr > 0)) {
                face_normal *= dr::rsqrt(length_sqr);
                
                // Vectorize side pairs to compute simultaneously
                auto side1 = transpose(dr::Array<dr::Packet<InputFloat, 3>, 3>{ side_0, triangle[2] - triangle[1], triangle[0] - triangle[2] });
                auto side2 = transpose(dr::Array<dr::Packet<InputFloat, 3>, 3>{ side_1, triangle[0] - triangle[1], triangle[1] - triangle[2] });
                InputVector3f face_angles = unit_angle(dr::normalize(side1), dr::normalize(side2));

                for (size_t j = 0; j < 3; ++j) {
                    vertex_normals[indices[j]] += face_normal * face_angles[j];
                }
            }
        }  
    }

    void update_vertex_normals_vectorized(Point3f* triangle, Vector3f& vertex_normals, Vector3u& indices)
    {
        if constexpr (dr::is_dynamic_v<Float>) {
            Vector3f face_normal = dr::normalize(dr::cross(triangle[1] - triangle[0], triangle[2] - triangle[0]));
    
            for (int i = 0; i < 3; ++i) {
                Vector3f d0 = dr::normalize(triangle[(i + 1) % 3] - triangle[i]);
                Vector3f d1 = dr::normalize(triangle[(i + 2) % 3] - triangle[i]);
                Float face_angle = dr::safe_acos(dr::dot(d0, d1));
                
                Vector3f nn = face_normal * face_angle;
                for (int j = 0; j < 3; ++j)
                    dr::scatter_reduce(ReduceOp::Add, vertex_normals[j], nn[j], indices[i]);
            }
        }
    }

    template <typename Index>
    MI_INLINE auto vertex_normal(Index index,
                                 dr::mask_t<Index> active = true) const {
        using Result = Normal<dr::replace_scalar_t<Index, InputFloat>, 3>;
        return dr::gather<Result>(m_vertex_normals, index, active);
    }


#if defined(MI_ENABLE_CUDA)
    using Base::m_optix_data_ptr;

    void optix_prepare_geometry() override {
        if constexpr (dr::is_cuda_v<Float>) {
            if (!m_optix_data_ptr)
                m_optix_data_ptr = jit_malloc(AllocType::Device, sizeof(OptixHeightfieldData));

            OptixHeightfieldData data =  { m_to_world.scalar(), m_to_object.scalar(), m_res_x, m_res_y, m_heightfield_texture.tensor().array().data(), m_max_height.scalar()};
            jit_memcpy(JitBackend::CUDA, m_optix_data_ptr, &data, sizeof(OptixHeightfieldData));
        }
    }

    void optix_build_input(OptixBuildInput &build_input) const override {
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build_input.customPrimitiveArray.aabbBuffers   = &m_device_bboxes;
        build_input.customPrimitiveArray.numPrimitives = m_amount_primitives;
        build_input.customPrimitiveArray.strideInBytes = 6 * sizeof(float);
        build_input.customPrimitiveArray.flags         = optix_geometry_flags;
        build_input.customPrimitiveArray.numSbtRecords = 1;
    }
#endif


#if defined(MI_ENABLE_CUDA)
    static constexpr uint32_t optix_geometry_flags[1] = {
        OPTIX_GEOMETRY_FLAG_NONE
    };
#endif

    bool parameters_grad_enabled() const override {
        bool result = false;
        result |= dr::grad_enabled(m_to_world);
        result |= dr::grad_enabled(m_max_height);
        result |= dr::grad_enabled(m_heightfield_texture); 

        return result;
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "Heightfield[" << std::endl
            << "  to_world = " << string::indent(m_to_world, 13) << "," << std::endl
            << "  max height = " << m_max_height << "," << std::endl
            << "  surface_area = " << surface_area() << "," << std::endl
            << "  " << string::indent(get_children_string()) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:

    // Heightfield data texture
    InputTexture2f m_heightfield_texture;

    // Resolution of heightfield texture in X and Y dimensions
    uint32_t m_res_x, m_res_y;

    // Max height displacement of each texel in the heightfield texture. A texel
    // with value 1.0 will be displaced by `m_max_height`, while a texel with value
    // 0.0 won't be displaced at all. 
    field<Float> m_max_height; 
    
    // Weak pointer to underlying grid texture data. Only used for llvm/scalar
    // variants. We store this because during raytracing, we don't want to call
    // Texture3f::tensor().data() which internally calls jit_var_ptr and is
    // guarded by a global state lock
    float *m_host_grid_data = nullptr;

    // Host-visible bounding boxes 
    // Also used when targeting CUDA variants, as we upload asynchronously
    // to GPU
    void *m_host_bboxes = nullptr;
    
    // Device-visible bounding boxes
    // Only valid for CUDA variants
    void *m_device_bboxes = nullptr;

    // Per vertex normals of triangulated heightfield
    FloatStorage m_vertex_normals; 
    bool m_has_vertex_normals;

    // The amount of vertices the heightfield consists of, this should be equal to
    // the resolution of the heightfield texture
    ScalarSize m_vertex_count = 0;

    size_t m_amount_primitives = 0;
};

MI_IMPLEMENT_CLASS_VARIANT(Heightfield, Shape)
MI_EXPORT_PLUGIN(Heightfield, "Heightfield intersection primitive");
NAMESPACE_END(mitsuba)