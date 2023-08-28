import mitsuba as mi
import drjit as dr
import numpy as np
import meshplot as mp

def get_mesh(heightfield_texture, res):
    vertices = np.empty([(res - 1) * (res - 1) * 6, 3])
    faces = np.empty([(res - 1) * (res - 1) * 2, 3])
    print(vertices.shape)

    cell_size = [2.0 / (res - 1), 2.0 / (res - 1)];
    amount_rows = res - 1
    face = 0
    # Loop over tiles
    for y in range(amount_rows):
        for x in range(amount_rows):
            local_min_bounds = [-1.0 + x * cell_size[0], -1.0 + y * cell_size[1]]
            local_max_bounds = [-1.0 + (x + 1) * cell_size[0], -1.0 + (y + 1) * cell_size[1]]

            left_bottom = (amount_rows - y) * res + x
            right_bottom = (amount_rows - y) * res + x + 1
            left_top = (amount_rows - (y + 1)) * res + x
            right_top = (amount_rows - (y + 1)) * res + x + 1
            
            tile_offset = (y * (res - 1) + x) * 6
            # Add vertices for each tile
            vertices[tile_offset + 0, :] = np.array([local_min_bounds[0], local_max_bounds[1], heightfield_texture[left_top]])  
            vertices[tile_offset + 1, :] = np.array([local_min_bounds[0], local_min_bounds[1], heightfield_texture[left_bottom]])  
            vertices[tile_offset + 2, :] = np.array([local_max_bounds[0], local_min_bounds[1], heightfield_texture[right_bottom]])  
            vertices[tile_offset + 3, :] = np.array([local_max_bounds[0], local_min_bounds[1], heightfield_texture[right_bottom]])  
            vertices[tile_offset + 4, :] = np.array([local_min_bounds[0], local_max_bounds[1], heightfield_texture[left_top]])  
            vertices[tile_offset + 5, :] = np.array([local_max_bounds[0], local_max_bounds[1], heightfield_texture[right_top]])  

            faces[face] = np.array([tile_offset, tile_offset + 1, tile_offset + 2])
            faces[face + 1] = np.array([tile_offset + 3, tile_offset + 4, tile_offset + 5])
            faces += 2
    return vertices, faces  

def main():
    mi.set_variant("scalar_rgb")
    scene = mi.load_file("../scenes/tests/ray_heightfield_intersect.xml")
    params = mi.traverse(scene)
    v, f = get_mesh(dr.ravel(params['Heightfield.heightfield']), 200)
    # data = np.load('data/data.npz')
    # v, f, n, fs = data["v"], data["f"], data["n"], data["fs"]

    # print(v)
    # print(f)

    mp.plot(v,f)

main()