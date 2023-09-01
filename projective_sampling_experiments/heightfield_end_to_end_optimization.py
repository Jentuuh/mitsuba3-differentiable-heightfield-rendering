#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import os 
import drjit as dr
import mitsuba as mi
import random
import numpy as np

mi.set_variant('cuda_ad_rgb')


# In[2]:


sensor_count = 8
sensor = {
    'type': 'batch',
    'film': {
        'type': 'hdrfilm',
        'width': 256 * sensor_count, 'height': 256,
        'filter': {'type': 'gaussian'},
        'sample_border': True
    }
}


# In[3]:


# Generate random viewpoints in upper hemispher
def random_upper_hemisphere(u, v, r):
    theta = u * (dr.pi / 4) + (dr.pi/4)
    phi = (v * dr.pi * 2) - dr.pi

    x = r * dr.sin(theta) * dr.cos(phi)
    y = r * dr.sin(theta) * dr.sin(phi)
    z = r * dr.cos(theta)
    return x, y, z


# In[4]:


from mitsuba import ScalarTransform4f as T

origins = []
for i in range(sensor_count):
    d = 4
    u = random.uniform(0, 1)
    v = random.uniform(0, 1)
    x_v, y_v, z_v = random_upper_hemisphere(u, v, d)

    origin = mi.ScalarPoint3f(x_v, y_v, z_v)
    origins.append(origin)

    sensor[f"sensor_{i}"] = {
        'type': 'perspective',
        'fov': 45,
        'to_world': T.look_at(target=[0, 0, 0], origin=origin, up=[0, 0, 1])
    }


# In[5]:


scene_dict_ref = {
    'type': 'scene',
    'integrator': {
        'type': 'direct_reparam',
    },
    'sensor': sensor,
    'sphere_2': {
        'type': 'sphere',
        'center': [0, 0, 5],
        'radius': 1,
       'emitter': {
            'type': 'area',
            'radiance': {
                'type': 'rgb',
                'value': 30.0,
            }
        }
    },
    'heightfield': {
        'type': 'heightfield',
        'filename': 'data/depth.bmp',
        'max_height': 1.0,
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.5, 0.5, 0.5]
            }
        }
    }
}

scene_target = mi.load_dict(scene_dict_ref)


# In[6]:


scene_dict_opt = {
    'type': 'scene',
    'integrator': {
        'type': 'direct_reparam',
    },
    'sensor': sensor,
    'sphere_2': {
        'type': 'sphere',
        'center': [0, 0, 5],
        'radius': 1,
       'emitter': {
            'type': 'area',
            'radiance': {
                'type': 'rgb',
                'value': 30.0,
            }
        }
    },
    'heightfield': {
        'type': 'heightfield',
        'resolution': 32,
        'max_height': 1.0,
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.5, 0.5, 0.5]
            }
        }
    }
}


# In[7]:


def plot_batch_output(out: mi.TensorXf):
    fig, ax = plt.subplots(figsize=(5*sensor_count, 5))
    ax.imshow(mi.util.convert_to_bitmap(out))
    ax.axis('off')


# In[8]:


ref_img = mi.render(scene_target, spp=256)
plot_batch_output(ref_img)


# In[9]:


scene_source = mi.load_dict(scene_dict_opt)

init_img = mi.render(scene_source, spp=128)
plot_batch_output(init_img)


# In[10]:


params = mi.traverse(scene_source)
res_x = params['heightfield.res_x']
res_y = params['heightfield.res_y']


# In[11]:


# opt = mi.ad.Adam(lr=0.02)
# opt['heightfield.heightfield'] = dr.clamp(params['heightfield.heightfield'], 0.0, 1.0) 

# for it in range(100):
#     loss = mi.Float(0.0)

#     params.update(opt)

#     img = mi.render(scene_source, params, seed=it, spp=16)

#     # L1 Loss
#     loss = dr.mean(dr.abs(img - ref_img))
#     dr.backward(loss)
#     opt.step()

#     print(f"Iteration {1+it:03d}: Loss = {loss[0]:6f}", end='\r')


# In[12]:


# final_img = mi.render(scene_source, spp=128)
# plot_batch_output(final_img)


# In[13]:


def scale_independent_loss(image, ref):
    """Brightness-independent L2 loss function."""
    scaled_image = image / dr.mean(dr.detach(image))
    scaled_ref = ref / dr.mean(ref)
    return dr.mean(dr.sqr(scaled_image - scaled_ref))


# In[14]:


def init_optimizer(lambda_):
    ls = mi.ad.LargeSteps(params['heightfield.heightfield'], lambda_)
    opt = mi.ad.Adam(lr=0.1, uniform=True)
    opt['u'] = ls.to_differential(params['heightfield.heightfield'], True)
    return ls, opt


# In[15]:


iterations = 1000
upsampling_steps = dr.sqr(dr.linspace(mi.Float, 0, 1, 3+1, endpoint=False).numpy()[1:])
upsampling_steps = (iterations * upsampling_steps).astype(int)
upsampling_steps


# In[16]:


lambda_ = 30
ls, opt = init_optimizer(lambda_)


# In[ ]:


iterations = 1000
for it in range(iterations):
    loss = mi.Float(0.0)
    
    if it in upsampling_steps:
        params['heightfield.heightfield'] = dr.upsample(params['heightfield.heightfield'], scale_factor=(2, 2, 1))
        params.update()
        res_x = params['heightfield.res_x']
        res_y = params['heightfield.res_y']
        lambda_ -= 1
        ls, opt = init_optimizer(lambda_)

    # Retrieve the vertex positions from the latent variable
    t = dr.unravel(mi.Point3f, ls.from_differential(opt['u']))
    params['heightfield.heightfield'] = mi.TensorXf(t.z, (res_x,res_y,1))
    params.update()

    img = mi.render(scene_source, params, seed=it, spp=64)
    mi.util.write_bitmap(f"output/iteration_{it}.png", img)
    
                                                # L1 Loss
    loss = scale_independent_loss(img, ref_img) #dr.mean(dr.abs(img - ref_img))
    dr.backward(loss)
    opt.step()

    print(f"Iteration {1+it:03d}: Loss = {loss[0]:6f}", end='\r')


# In[ ]:


# Update the mesh after the last iteration's gradient step
t = dr.unravel(mi.Point3f, ls.from_differential(opt['u']))
params['heightfield.heightfield'] = mi.TensorXf(t.z, (res_x,res_y,1))
params.update();


# In[ ]:


final_img = mi.render(scene_source, spp=128)
plot_batch_output(final_img)


# In[ ]:




