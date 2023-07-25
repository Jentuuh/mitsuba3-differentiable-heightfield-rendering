import pytest
import drjit as dr
import mitsuba as mi

from drjit.scalar import ArrayXf as Float

def test01_create(variant_scalar_rgb):
    s = mi.load_dict({
        "type": "heightfield",
        "max_height": 2.0,
    })
    assert s is not None


def test02_bbox(variant_scalar_rgb):
    pytest.importorskip("numpy")
    import numpy as np
    heightfield = np.array([
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0
    ]).reshape((4, 4, 1))

    # Test 01: Tests bbox encapsulating entire heightfield
    sy = 2.5
    for sx in [1, 2, 4]:
        for h in [5.0, 1.0, 10.0]:
            for translate in [mi.Vector3f([1.3, -3.0, 5]),
                            mi.Vector3f([-10000, 3.0, 31])]:
                s = mi.load_dict({
                    "type" : "scene",
                    "heightfield": {
                        "type" : "heightfield",
                        "to_world" : mi.Transform4f.translate(translate) @ mi.Transform4f.scale((sx, sy, 1.0)),
                        "max_height": h
                    }
                })
                params = mi.traverse(s)
                params['heightfield.heightfield'] = mi.TensorXf(heightfield)
                params.update()

                b = s.bbox()

                assert b.valid()
                assert dr.allclose(b.center(), translate + mi.ScalarVector3f([0.0, 0.0, 0.5 * h]))
                assert dr.allclose(b.min, translate - [sx, sy, 0])
                assert dr.allclose(b.max, translate + [sx, sy, h])
    # TODO: Test 02: Tests bboxs made for each texel of the heightfield 



def test03_parameters_changed(variant_scalar_rgb):

    assert True

def test04_ray_intersect(variant_scalar_rgb):
    pytest.importorskip("numpy")
    import numpy as np

    # ------------------------
    # |  0  |  0  |  0  |  0 |
    # |----------------------|
    # | 0   | .25 | .5  |  0 |
    # |----------------------|
    # | 0   | .25 | .5  | 0  |           
    # |----------------------|
    # | 0   | 0   | 0   | 0  |
    # ------------------------
    heightfield = np.array([
        0, 0, 0, 0,
        0, 0.25, 0.5, 0,
        0, 0.25, 0.5, 0,
        0, 0, 0, 0
    ]).reshape((4, 4, 1))


    for translate in [mi.ScalarVector3f([0.0, 0.0, 0.0])]:

        s = mi.load_dict({
                    "type" : "scene",
                    "heightfield": {
                        "type" : "heightfield",
                        "max_height" : 1.0,
                        "to_world" : mi.ScalarTransform4f.translate(translate),
                        # "filename": "../../../scenes/tests/data/test.bmp"

                    }
                })
        params = mi.traverse(s)
        params['heightfield.heightfield'] = mi.TensorXf(heightfield)
        params.update()

        n = 15
        x_values = dr.linspace(Float, -0.99, 0.99, n)
        rays = [mi.Ray3f(o=mi.Vector3f(x, 0.0, 5), d=mi.Vector3f(0, 0, -1)) for x in x_values]
        for i in range(n):

            si_found = s.ray_test(rays[i])
            si = s.ray_intersect(rays[i])

            # print(si_found, (x_values[i] >= 0.0 and heights[i] <= 0.25))
            
            print(rays[i])
            print(si.t, si.p.z)
            assert si_found == ((x_values[i] >= -1.0 and x_values[i] <= -0.33 and dr.allclose(si.p.z, 0.25))
                                or (x_values[i] >= -0.33 and x_values[i] <= 0.33 and dr.allclose(si.p.z, 0.5))
                                or (x_values[i] >= 0.33 and x_values[i] <= 1.0 and dr.allclose(si.p.z, 0.5)) )
            assert si.is_valid() == si_found


# def test05_ray_intersect_instancing(variants_all_ad_rgb):

#     assert True



