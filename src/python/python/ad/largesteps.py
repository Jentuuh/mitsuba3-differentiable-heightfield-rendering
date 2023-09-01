from __future__ import annotations as __annotations__ # Delayed parsing of type annotations

import mitsuba as mi
import drjit as dr

def mesh_laplacian(n_verts, faces, lambda_):
    """
    Compute the index and data arrays of the (combinatorial) Laplacian matrix of
    a given mesh.
    """
    import numpy as np

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = np.unique(np.stack([np.concatenate([ii, jj]), np.concatenate([jj, ii])], axis=0), axis=1)
    adj_values = np.ones(adj.shape[1], dtype=np.float64) * lambda_

    # Diagonal indices, duplicated as many times as the connectivity of each index
    diag_idx = np.stack((adj[0], adj[0]), axis=0)

    diag = np.stack((np.arange(n_verts), np.arange(n_verts)), axis=0)

    # Build the sparse matrix
    idx = np.concatenate((adj, diag_idx, diag), axis=1)
    values = np.concatenate((-adj_values, adj_values, np.ones(n_verts)))

    return idx, values

def heightfield_indices(res_x, res_y):
    import numpy as np

    indices = np.zeros((res_x - 1) * (res_y - 1) * 6)

    curr_i_idx = 0
    # Walk over heightfield tiles, add face indices for the 2 triangles of each tile
    for row_nr in range(res_y - 1):
        for row_offset in range(res_x - 1):
            # ===============================
            #       Build index buffer
            # ===============================
            # Triangle 1
            indices[curr_i_idx] = (row_nr * res_x) + row_offset
            indices[curr_i_idx + 1] = ((row_nr + 1) * res_x) + row_offset
            indices[curr_i_idx + 2] = ((row_nr + 1) * res_x) + row_offset + 1
            curr_i_idx += 3

            # Triangle 2
            indices[curr_i_idx] = (row_nr * res_x) + row_offset + 1
            indices[curr_i_idx + 1] = (row_nr * res_x) + row_offset
            indices[curr_i_idx + 2] = ((row_nr + 1) * res_x) + row_offset + 1
            curr_i_idx += 3

    return indices

def heightfield_vertices(heightfield_texture, res_x, res_y, max_height):
    # import numpy as np
    heightfield_flat = dr.ravel(heightfield_texture)
    vertices = dr.zeros(mi.Float, (res_x * res_y * 3))

    axis_offset = [2.0 / res_x, 2.0 / res_y]
    abs_index = 0
    tex_index = 0
    # Loop over tiles
    for y in range(res_y):
        for x in range(res_x):
            x_pos = -1.0 + (x * axis_offset[0])
            y_pos = 1.0 - (y * axis_offset[1])
            z_pos = heightfield_flat[tex_index] * max_height

            vertices[abs_index] = x_pos
            vertices[abs_index + 1] = y_pos
            vertices[abs_index + 2] = z_pos
            abs_index += 3
            tex_index += 1

    return vertices  


class SolveCholesky(dr.CustomOp):
    """
    DrJIT custom operator to solve a linear system using a Cholesky factorization.
    """

    def eval(self, solver, u):
        self.solver = solver
        x = dr.empty(mi.TensorXf, shape=u.shape)
        solver.solve(u, x)
        return mi.TensorXf(x)

    def forward(self):
        x = dr.empty(mi.TensorXf, shape=self.grad_in('u').shape)
        self.solver.solve(self.grad_in('u'), x)
        self.set_grad_out(x)

    def backward(self):
        x = dr.empty(mi.TensorXf, shape=self.grad_out().shape)
        self.solver.solve(self.grad_out(), x)
        self.set_grad_in('u', x)

    def name(self):
        return "Cholesky solve"


class LargeSteps():
    """
    Implementation of the algorithm described in the paper "Large Steps in
    Inverse Rendering of Geometry" (Nicolet et al. 2021).

    It consists in computing a latent variable u = (I + λL) v from the vertex
    positions v, where L is the (combinatorial) Laplacian matrix of the input
    mesh. Optimizing these variables instead of the vertex positions allows to
    diffuse gradients on the surface, which helps fight their sparsity.

    This class builds the system matrix (I + λL) for a given mesh and hyper
    parameter λ, and computes its Cholesky factorization.

    It can then convert vertex coordinates back and forth between their
    cartesian and differential representations. Both transformations are
    differentiable, meshes can therefore be optimized by using the differential
    form as a latent variable.
    """
    def __init__(self, verts, faces, lambda_=19.0):
        """
        Build the system matrix and its Cholesky factorization.

        Parameter ``verts`` (``mitsuba.Float``):
            Vertex coordinates of the mesh.

        Parameter ``faces`` (``mitsuba.UInt``):
            Face indices of the mesh.

        Parameter ``lambda_`` (``float``):
            The hyper parameter λ. This controls how much gradients are diffused
            on the surface. this value should increase with the tesselation of
            the mesh.

        """
        if mi.variant().endswith('double'):
            from cholespy import CholeskySolverD as CholeskySolver
        else:
            from cholespy import CholeskySolverF as CholeskySolver

        from cholespy import MatrixType
        import numpy as np

        v = verts.numpy().reshape((-1,3))
        f = faces.numpy().reshape((-1,3))

        # Remove duplicates due to e.g. UV seams or face normals.
        # This is necessary to avoid seams opening up during optimisation
        v_unique, index_v, inverse_v = np.unique(v, return_index=True, return_inverse=True, axis=0)
        f_unique = inverse_v[f]

        self.index = mi.UInt(index_v)
        self.inverse = mi.UInt(inverse_v)
        self.n_verts = v_unique.shape[0]

        # Solver expects matrices without duplicate entries as input, so we need to sum them manually
        indices, values = mesh_laplacian(self.n_verts, f_unique, lambda_)
        indices_unique, inverse_idx = np.unique(indices, axis=1, return_inverse=True)

        self.rows = mi.TensorXi(indices_unique[0])
        self.cols = mi.TensorXi(indices_unique[1])
        data = dr.zeros(mi.TensorXd, shape=(indices_unique.shape[1],))

        dr.scatter_reduce(dr.ReduceOp.Add, data.array, mi.Float64(values), mi.UInt(inverse_idx))

        self.solver = CholeskySolver(self.n_verts, self.rows, self.cols, data, MatrixType.COO)
        self.data = mi.TensorXf(data)

    
    def __init__(self, heightfield_tensor, lambda_=19.0):
        """
        Build the system matrix and its Cholesky factorization. (For heightfield shapes)

        Parameter ``heightfield_texture`` (``mitsuba.Float``):
            Tensor that contains the heightfield data.

        Parameter ``lambda_`` (``float``):
            The hyper parameter λ. This controls how much gradients are diffused
            on the surface. this value should increase with the tesselation of
            the mesh.
        """
        if mi.variant().endswith('double'):
            from cholespy import CholeskySolverD as CholeskySolver
        else:
            from cholespy import CholeskySolverF as CholeskySolver

        from cholespy import MatrixType
        import numpy as np

        index_buffer = heightfield_indices(heightfield_tensor.shape[0], heightfield_tensor.shape[1])
        face_buffer = index_buffer.reshape((-1,3))
        self.index = np.arange(0, heightfield_tensor.shape[0] * heightfield_tensor.shape[1], 1.0) # For the heightfield, we start from unique vertices, so no remapping is necessary
        self.inverse = np.arange(0, heightfield_tensor.shape[0] * heightfield_tensor.shape[1], 1.0) 

        # Amount of unique vertices is equal to resolution of heightfield texture
        self.n_verts = heightfield_tensor.shape[0] * heightfield_tensor.shape[1]

        # Solver expects matrices without duplicate entries as input, so we need to sum them manually
        indices, values = mesh_laplacian(self.n_verts, face_buffer, lambda_)
        indices_unique, inverse_idx = np.unique(indices, axis=1, return_inverse=True)

        self.rows = mi.TensorXi(indices_unique[0])
        self.cols = mi.TensorXi(indices_unique[1])
        data = dr.zeros(mi.TensorXd, shape=(indices_unique.shape[1],))

        dr.scatter_reduce(dr.ReduceOp.Add, data.array, mi.Float64(values), mi.UInt(inverse_idx))

        self.solver = CholeskySolver(self.n_verts, self.rows, self.cols, data, MatrixType.COO)
        self.data = mi.TensorXf(data)


    def to_differential(self, v, usage_heightfield = False):
        """
        Convert vertex coordinates to their differential form: u = (I + λL) v.

        This method typically only needs to be called once per mesh, to obtain
        the latent variable before optimization.

        Parameter ``v`` (``mitsuba.Float``):
            Vertex coordinates of the mesh.

        Returns ``mitsuba.Float`:
            Differential form of v.
        """
        if usage_heightfield:
            v = heightfield_vertices(v, v.shape[0], v.shape[1], 1.0)
        
        # Manual matrix-vector multiplication
        v_unique = dr.gather(mi.Point3f, dr.unravel(mi.Point3f, v), self.index)
        row_prod = dr.gather(mi.Point3f, v_unique, self.cols.array) * self.data.array
        u = dr.zeros(mi.Point3f, dr.width(v_unique))
        dr.scatter_reduce(dr.ReduceOp.Add, u, row_prod, self.rows.array)
    
        return dr.ravel(u)

    def from_differential(self, u):
        """
        Convert differential coordinates back to their cartesian form: v = (I +
        λL)⁻¹ u.

        This is done by solving the linear system (I + λL) v = u using the
        previously computed Cholesky factorization.

        This method is typically called at each iteration of the optimization,
        to update the mesh coordinates before rendering.

        Parameter ``u`` (``mitsuba.Float``):
            Differential form of v.

        Returns ``mitsuba.Float`:
            Vertex coordinates of the mesh.
        """
        v_unique = dr.unravel(mi.Point3f, dr.custom(SolveCholesky, self.solver, mi.TensorXf(u, (self.n_verts, 3))).array)
        return dr.ravel(dr.gather(mi.Point3f, v_unique, self.inverse))
