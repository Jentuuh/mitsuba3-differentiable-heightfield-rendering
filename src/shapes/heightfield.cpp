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
   - Name of the file that stores the heightfield.
   - |exposed|

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

    using typename Base::ScalarIndex;
    using typename Base::ScalarSize;

    Heightfield(const Properties &props) : Base(props) {        
        // Maximal height value of the heightfield (normalized height values will be scaled accordingly)
        m_max_height = props.get<ScalarFloat>("max_height", 1.0f);

        // Load heightfield data into Bitmap object
        if (props.has_property("filename")) {
            FileResolver *fs   = Thread::thread()->file_resolver();
            fs::path file_path = fs->resolve(props.string("filename"));
            if (!fs::exists(file_path))
                Log(Error, "\"%s\": file does not exist!", file_path);
            ref<Bitmap> heightfield_bitmap = new Bitmap(file_path);
            
            // Convert to float32 representation
            ref<Bitmap> normalized = heightfield_bitmap->convert(Bitmap::PixelFormat::Y, Struct::Type::Float32, false);
            //print_heightfield_values(normalized);

            m_res_x = normalized->width();
            m_res_y = normalized->height();

            // --!-- Tensor dimension should be texture dimension + 1 (for color channels) --!--
            size_t shape[3] = {m_res_x, m_res_y, 1};
            
            m_heightfield_texture = InputTexture2f(InputTensorXf((float*)normalized->data(), 3, shape), true, false,
                dr::FilterMode::Linear, dr::WrapMode::Clamp);        
        } else {
            InputFloat default_data[16] = { 0.f, 0.f, 0.f, 0.f,
                                           0.f, 0.f, 0.f, 0.f, 
                                           0.f, 0.f, 0.f, 0.f,
                                           0.f, 0.f, 0.f, 0.f };

            m_res_x = 4;
            m_res_y = 4;

            size_t default_shape[3] = { m_res_x, m_res_y, 1 };
            m_heightfield_texture = InputTexture2f(
                InputTensorXf(default_data, 3, default_shape), true, false,
                dr::FilterMode::Linear, dr::WrapMode::Clamp);
        }
        update();
        initialize();
    }

    ~Heightfield() {
        jit_free(m_host_bboxes);
        jit_free(m_device_bboxes);
    }

    void update() {
        auto [S, Q, T] =
            dr::transform_decompose(m_to_world.scalar().matrix, 25);
        if (dr::abs(Q[0]) > 1e-6f || dr::abs(Q[1]) > 1e-6f ||
            dr::abs(Q[2]) > 1e-6f || dr::abs(Q[3] - 1) > 1e-6f)
            Log(Warn, "'to_world' transform shouldn't perform any rotations, "
                      "use instancing (`shapegroup` and `instance` plugins) "
                      "instead!");

        Vector3f dp_du = m_to_world.value() * Vector3f(2.f, 0.f, 0.f);
        Vector3f dp_dv = m_to_world.value() * Vector3f(0.f, 2.f, 0.f);
        Normal3f normal = dr::normalize(m_to_world.value() * Normal3f(0.f, 0.f, 1.f));
        m_frame = Frame3f(dp_du, dp_dv, normal);

        m_to_object = m_to_world.value().inverse();

        if constexpr (!dr::is_cuda_v<Float>) {
            m_host_grid_data = m_heightfield_texture.tensor().data();
        }

        jit_free(m_host_bboxes);
        jit_free(m_device_bboxes);
        std::tie(m_host_bboxes,
                 m_device_bboxes,
                 m_amount_primitives) = build_bboxes();

        mark_dirty();
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_parameter("to_world", *m_to_world.ptr(), ParamFlags::Differentiable | ParamFlags::Discontinuous);
        callback->put_parameter("heightfield", m_heightfield_texture.tensor(), ParamFlags::Differentiable | ParamFlags::Discontinuous);
    }

    void parameters_changed(const std::vector<std::string> &keys) override {
        if (keys.empty() || string::contains(keys, "to_world") || string::contains(keys, "heightfield_data")) {
            // Ensure previous ray-tracing operation are fully evaluated before
            // modifying the scalar values of the fields in this class
            if constexpr (dr::is_jit_v<Float>)
                dr::sync_thread();

            // Update the scalar value of the matrix
            m_to_world = m_to_world.value();

            // Update heightfield texture
            m_heightfield_texture.set_tensor(m_heightfield_texture.tensor());
            update();
        }

        Base::parameters_changed();
    }


    /* \brief Computes AABBs for all heightfield cells (a heightfield cell 
     * implicitly contains a surface inside). Returns a pointer to the array 
     * of AABBs, a pointer to an array of cell indices of the former AABBs and the
     * amount of AABBs that were initialized.
     */
    std::tuple<void *, void *, size_t> build_bboxes() {
        auto shape = m_heightfield_texture.tensor().shape();    
        float cell_size[2] = { 2.0f / (m_res_x - 1), 2.0f / (m_res_y - 1)};
        
        // Bbox count is shape[d] - 1 because our tensor data represents heightfield values at cell corners
        // (4 heightfield texels form 1 heightfield cell)
        size_t max_bbox_count =
            (shape[0] - 1) * (shape[1] - 1);
        ScalarTransform4f to_world = m_to_world.scalar();

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

        size_t count = 0;
        // Loop over heightfield texels and build AABB for each
        for (size_t y = shape[0] - 1; y > 0; --y) {
            for (size_t x = 0; x < shape[1] - 1; ++x) {
                
                // TODO: unreadable point computation code, see how we can rewrite this

                ScalarBoundingBox3f bbox;
                // 00 (left bottom)
                bbox.expand(to_world.transform_affine(ScalarPoint3f(       
                    -1.0f + ((x + 0) * cell_size[1]), - 1.0f + (((shape[0] - 1) - (y - 0)) * cell_size[0]), 
                        m_max_height.scalar() * grid[(y - 0) * shape[0] + (x + 0)])));
                // 01 (left top)
                bbox.expand(to_world.transform_affine(ScalarPoint3f(
                        -1.0f + ((x + 0) * cell_size[1]), -1.0f + (((shape[0] - 1) - (y - 1)) * cell_size[0]), 
                        m_max_height.scalar() * grid[(y - 1) * shape[0] + (x + 0)])));
                // 10 (right bottom)
                bbox.expand(to_world.transform_affine(ScalarPoint3f(
                        -1.0f + ((x + 1) * cell_size[1]), -1.0f + (((shape[0] - 1) - (y - 0)) * cell_size[0]), 
                        m_max_height.scalar() * grid[(y - 0) * shape[0] + (x + 1)])));
                // 11 (right top)
                bbox.expand(to_world.transform_affine(ScalarPoint3f(
                        -1.0f + ((x + 1) * cell_size[1]), -1.0f + (((shape[0] - 1) - (y - 1)) * cell_size[0]), 
                        m_max_height.scalar() * grid[(y - 1) * shape[0] + (x + 1)])));

                host_aabbs[count] = BoundingBoxType(bbox);
                count++;
            }
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
        bbox.expand(to_world.transform_affine(ScalarPoint3f( 0.f,  0.f, m_max_height.scalar())));
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
        
        float cell_size[2] = { 2.0f / (m_res_x - 1), 2.0f / (m_res_y - 1)};

        uint32_t amount_rows = m_res_x - 1;
        uint32_t values_per_row = m_res_x;
        uint32_t row_nr = dr::floor((float)prim_index / (float) amount_rows); // floor(prim_index / amount_bboxes_per_row)
        uint32_t row_offset = prim_index % (amount_rows); // prim_index % amount_bboxes_per_row

        // Compute the fractional bounds of the cell we're testing the intersection for
        Point<FloatP, 2> local_min_target_bounds = Point<FloatP, 2>(-1.0f + (float)row_offset * cell_size[0], -1.0f + (float)row_nr * cell_size[1]);
        Point<FloatP, 2> local_max_target_bounds = Point<FloatP, 2>(-1.0f + (float)(row_offset + 1) * cell_size[0], -1.0f + (float)(row_nr + 1) * cell_size[1]);

        // Corresponds to rectangle intersection, except that each voxel now has its own small rectangle 
        // to whose space we should transform the ray.
        Ray3fP ray;

        // Transform ray to local space
        if constexpr (!dr::is_jit_v<FloatP>)
            ray = m_to_object.scalar().transform_affine(ray_);
        else
            ray = m_to_object.value().transform_affine(ray_);
        
        // `(amount_rows - row_nr) * values_per_row` gives us the offset to get to the current row, we add the 
        // row offset to this to get the absolute offset to obtain the corresponding texel of the
        // current AABB (+ 1 in both dimensions to get right and top texels)  
        uint32_t left_bottom_index = (amount_rows - row_nr) * values_per_row + row_offset;
        uint32_t right_bottom_index = (amount_rows - row_nr) * values_per_row + row_offset + 1;
        uint32_t left_top_index = (amount_rows - (row_nr + 1)) * values_per_row + row_offset;
        uint32_t right_top_index = (amount_rows - (row_nr + 1)) * values_per_row + row_offset + 1;

        FloatP max_displacement_in_tile = dr::maximum(m_heightfield_texture.tensor().data()[left_bottom_index],
                                                   dr::maximum(m_heightfield_texture.tensor().data()[right_bottom_index],
                                                   dr::maximum(m_heightfield_texture.tensor().data()[left_top_index],
                                                   m_heightfield_texture.tensor().data()[right_top_index])));

        // Check how high the heightfield is at current cell (we will intersect a plane at this height parallel to the XY plane)
        FloatP z_displacement = max_displacement_in_tile * m_max_height.scalar();
        
        // We intersect with the plane Z = `z_displacement`, parallel to the XY plane (flat heightfield in local space is defined as a rectangle aligned with XY)
        FloatP t = (z_displacement - ray.o.z()) / ray.d.z();
        Point<FloatP, 3> local = ray(t);

        // Is intersection within ray segment and heightfield cell?
        active = active && t >= 0.f
                        && t <= ray.maxt
                        && local.x() >= local_min_target_bounds.x()
                        && local.y() >= local_min_target_bounds.y()
                        && local.x() <= local_max_target_bounds.x()
                        && local.y() <= local_max_target_bounds.y();

        
        return { dr::select(active, t, dr::Infinity<FloatP>),
            Point<FloatP, 2>(local.x(), local.y()), ((uint32_t) -1), prim_index };
    }


    template <typename FloatP, typename Ray3fP>
    dr::mask_t<FloatP> ray_test_impl(const Ray3fP &ray_,
                                     ScalarIndex prim_index,
                                     dr::mask_t<FloatP> active) const {
        MI_MASK_ARGUMENT(active);    
             float cell_size[2] = { 2.0f / (m_res_x - 1), 2.0f / (m_res_y - 1)};

        uint32_t amount_rows = m_res_x - 1;
        uint32_t values_per_row = m_res_x;
        uint32_t row_nr = dr::floor((float)prim_index / (float)(values_per_row - 1)); // floor(prim_index / amount_bboxes_per_row)
        uint32_t row_offset = prim_index % (values_per_row - 1); // prim_index % amount_bboxes_per_row

        // Compute the fractional bounds of the cell we're testing the intersection for
        Point<FloatP, 2> local_min_target_bounds = Point<FloatP, 2>(-1.0f + (float)row_offset * cell_size[0], -1.0f + (float)row_nr * cell_size[1]);
        Point<FloatP, 2> local_max_target_bounds = Point<FloatP, 2>(-1.0f + (float)(row_offset + 1) * cell_size[0], -1.0f + (float)(row_nr + 1) * cell_size[1]);

        // Corresponds to rectangle intersection, except that each voxel now has its own rectangle 
        // to whose space we should transform the ray.
        Ray3fP ray;

        // Transform ray to local space
        if constexpr (!dr::is_jit_v<FloatP>)
            ray = m_to_object.scalar().transform_affine(ray_);
        else
            ray = m_to_object.value().transform_affine(ray_);
        
        // `row_nr * values_per_row` gives us the offset to get to the current row, we add the 
        // row offset to this to get the absolute offset to obtain the corresponding texel of the
        // current AABB (+ 1 in both dimensions to get right and top texels)  
        uint32_t left_bottom_index = (amount_rows - row_nr) * values_per_row + row_offset;
        uint32_t right_bottom_index = (amount_rows - row_nr) * values_per_row + row_offset + 1;
        uint32_t left_top_index = (amount_rows - (row_nr + 1)) * values_per_row + row_offset;
        uint32_t right_top_index = (amount_rows - (row_nr + 1)) * values_per_row + row_offset + 1;

        FloatP max_displacement_in_tile = dr::maximum(m_heightfield_texture.tensor().data()[left_bottom_index],
                                                   dr::maximum(m_heightfield_texture.tensor().data()[right_bottom_index],
                                                   dr::maximum(m_heightfield_texture.tensor().data()[left_top_index],
                                                   m_heightfield_texture.tensor().data()[right_top_index])));

        // Check how high the heightfield is at current cell (we will intersect a plane at this height parallel to the XY plane)
        FloatP z_displacement = max_displacement_in_tile * m_max_height.scalar();
        
        // We intersect with the plane Z = `z_displacement`, parallel to the XY plane (flat heightfield in local space is defined as a rectangle aligned with XY)
        FloatP t = (z_displacement - ray.o.z()) / ray.d.z();
        Point<FloatP, 3> local = ray(t);

        // Is intersection within ray segment and heightfield cell?
        return active && t >= 0.f
                        && t <= ray.maxt
                        && local.x() >= local_min_target_bounds.x()
                        && local.y() >= local_min_target_bounds.y()
                        && local.x() <= local_max_target_bounds.x()
                        && local.y() <= local_max_target_bounds.y();
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
        
        Transform4f to_world = m_to_world.value();
        Transform4f to_object = m_to_object.value();

        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();

        // if constexpr (IsDiff) {

        // } else {
        si.t = pi.t;
        si.p = ray(pi.t);
            // // Re-project intersection point found along ray onto the heightfield to improve accuracy
            // Point3f p = ray(pi.t);
            // Float dist = dr::dot(to_world.translation() - p, m_frame.n);
            // si.p = p + dist * m_frame.n;
        // }

        si.t = dr::select(active, si.t, dr::Infinity<Float>);
        si.n          = m_frame.n;
        si.sh_frame.n = m_frame.n;
        si.dp_du      = m_frame.s;
        si.dp_dv      = m_frame.t;
        
        si.dn_du = si.dn_dv = dr::zeros<Vector3f>();
        si.shape    = this;
        si.instance = nullptr;

        return si;
    }
    
    // ==========================================================================
    // Debugging helper (Remove later)
    // ==========================================================================
    MI_INLINE void print_heightfield_values(ref<Bitmap> heightfield_data) const {
        uint32_t width = heightfield_data->width();
        uint32_t height = heightfield_data->height();
        for(uint32_t x = 0; x < width; x++)
        {
            std::cout << "------------------------------------------------" << std::endl;
            for(uint32_t y = 0; y < height; y++)
            {
                std::cout <<"| " << ((float*)heightfield_data->data())[x * width + y] << " |";
            }
        }
        std::cout << "------------------------------------------------" << std::endl;

    }

    MI_INLINE void print_heightfield_texture() const {
        for(uint32_t x = 0; x < m_res_x; x++)
        {
            std::cout << "------------------------------------------------" << std::endl;
            for(uint32_t y = 0; y < m_res_y; y++)
            {
                std::cout <<"| " << (m_heightfield_texture.tensor().data())[x * m_res_x + y] << " |";
            }
        }
        std::cout << "------------------------------------------------" << std::endl;

    }
    // ==============================================================================

#if defined(MI_ENABLE_CUDA)
    using Base::m_optix_data_ptr;

    void optix_prepare_geometry() override {
        if constexpr (dr::is_cuda_v<Float>) {
            if (!m_optix_data_ptr)
                m_optix_data_ptr = jit_malloc(AllocType::Device, sizeof(OptixHeightfieldData));

            OptixHeightfieldData data =  { m_to_object.scalar(), m_res_x, m_res_y, m_heightfield_texture.tensor().array().data(), m_max_height.scalar()};
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
        return dr::grad_enabled(m_to_world);
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
    size_t m_res_x, m_res_y;

    // Max height displacement of each texel in the heightfield texture. A texel
    // with value 1.0 will be displaced by `m_max_height`, while a texel with value
    // 0.0 won't be displaced at all. 
    field<Float> m_max_height; 
    
    Frame3f m_frame;

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

    size_t m_amount_primitives = 0;
};

MI_IMPLEMENT_CLASS_VARIANT(Heightfield, Shape)
MI_EXPORT_PLUGIN(Heightfield, "Heightfield intersection primitive");
NAMESPACE_END(mitsuba)