import torch, cv2
import trimesh, trimesh.ray
import numpy as np
from utils import fileio
from numba import jit
from scipy.spatial import Delaunay
import pymeshlab


@jit(nopython=True)
def remove_face_by_point_idx(face_array_matrix, indicator):
    face_array_matrix_crop = []
    lenth = face_array_matrix.shape[0]
    for i in range(lenth):
        if indicator[face_array_matrix[i][0]] and \
                indicator[face_array_matrix[i][1]] and \
                indicator[face_array_matrix[i][2]]:
            face_array_matrix_crop.append(face_array_matrix[i])
    return face_array_matrix_crop


def crop_spherically(pcl, center, r):
    """
    :param pcl: n*3
    :param center: [x,y,z]
    :param r: range
    :return:
    """
    n = pcl.shape[0]
    center = np.repeat(center[np.newaxis, :], n, axis=0).reshape((n, 3))
    dis = pcl - center
    dis = (dis * dis).sum(axis=1)
    return pcl[np.where(dis < r * r)], np.where(dis < r * r)


def sample_uniform_points_in_sphere(amount, radius=1.):
    sphere_points = np.random.uniform(-radius, radius, size=(amount * 2 + 20, 3))
    sphere_points = sphere_points[np.linalg.norm(sphere_points, axis=1) < radius]

    points_available = sphere_points.shape[0]
    if points_available < amount:
        # This is a fallback for the rare case that too few points are inside the unit sphere
        result = np.zeros((amount, 3))
        result[:points_available, :] = sphere_points
        result[points_available:, :] = sample_uniform_points_in_sphere(amount - points_available, radius=radius)
        return result
    else:
        return sphere_points[:amount, :]


def check_ray_triangle_intersection(ray_origins, ray_direction, triangle, epsilon=1e-6):
    """
    Optimized to work for:
        >1 ray_origins
        1 ray_direction multiplied to match the dimension of ray_origins
        1 triangle
    Based on: Answer by BrunoLevy at
    https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d
    Thank you!
    Parameters
    ----------
    ray_origin : torch.Tensor, (n_rays, n_dimensions), (x, 3)
    ray_directions : torch.Tensor, (n_rays, n_dimensions), (1, x)
    triangle : torch.Tensor, (n_points, n_dimensions), (3, 3)
    Return
    ------
    intersection : boolean (n_rays,)
    Test
    ----
    triangle = torch.Tensor([[0., 0., 0.],
                             [1., 0., 0.],
                             [0., 1., 0.],
                            ]).to(device)
    ray_origins = torch.Tensor([[0.5, 0.25, 0.25],
                                [5.0, 0.25, 0.25],
                               ]).to(device)
    ray_origins = torch.rand((10000, 3)).to(device)
    ray_direction = torch.Tensor([[0., 0., -10.],]).to(device)
    #ray_direction = torch.Tensor([[0., 0., 10.],]).to(device)
    ray_direction = ray_directions.repeat(ray_origins.shape[0], 1)
    check_ray_triangle_intersection(ray_origins, ray_direction, triangle)
    """

    E1 = triangle[1] - triangle[0]  # vector of edge 1 on triangle
    E2 = triangle[2] - triangle[0]  # vector of edge 2 on triangle
    N = torch.cross(E1, E2)  # normal to E1 and E2

    invdet = 1. / -torch.einsum('ji, i -> j', ray_direction, N)  # inverse determinant

    A0 = ray_origins - triangle[0]
    # print('A0.shape: ', A0.shape)
    # print('ray_direction.shape: ', ray_direction.shape)
    DA0 = torch.cross(A0, ray_direction.repeat(A0.size(0), 1), dim=1)

    u = torch.einsum('ji, i -> j', DA0, E2) * invdet
    v = -torch.einsum('ji, i -> j', DA0, E1) * invdet
    t = torch.einsum('ji, i -> j', A0, N) * invdet

    intersection = (t >= 0.0) * (u >= 0.0) * (v >= 0.0) * ((u + v) <= 1.0)

    return intersection


def hidden_surface_remove(vertices, triangle_idx, direction=None, epsilon=1e-6):
    if direction is None:
        direction = [0., 0., -1.]
    direction = -np.array(direction)
    vertices, triangle_idx = np.array(vertices, dtype=float), np.array(triangle_idx, dtype=int)
    vn, fn = vertices.shape[0], triangle_idx.shape[0]
    triangles = vertices[triangle_idx]
    ray_directions = np.repeat(direction[np.newaxis, :], vn, axis=0)
    index_triangle, index_ray, locations = trimesh.ray.ray_triangle.ray_triangle_id(triangles, vertices, ray_directions)
    hidden_idx = np.where(np.linalg.norm(vertices[index_ray] - locations, axis=1) > epsilon)[0]
    hidden_idx = np.unique(index_ray[hidden_idx])

    mask = np.ones(vn)
    mask[hidden_idx] = 0
    return mask


def remove_vertices_from_mesh(vertices, triangle_idx, mask):
    """
    :param vertices: (n, 3) float
    :param triangle_idx: (m, 3) int
    :param mask: (n,) where 1 indicates preserve
    :return: vertices, triangle_idx after remove
    """
    indicator = mask
    ms = pymeshlab.MeshSet()
    face_array_matrix_crop = remove_face_by_point_idx(triangle_idx, indicator)
    face_array_matrix_crop = np.array(face_array_matrix_crop)

    tri_crop = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=face_array_matrix_crop)
    ms.add_mesh(tri_crop)
    ms.remove_unreferenced_vertices()
    result_mesh = ms.current_mesh()

    return result_mesh.vertex_matrix(), result_mesh.face_matrix()


def rays_triangles_intersection(ray_origins, ray_direction, triangle, epsilon=1e-6):
    """

    :param ray_origins: torch.Tensor, (n_rays, n_dimensions), (n, 3)
    :param ray_direction:  torch.Tensor, (n_rays, n_dimensions), (n, 3)
    :param triangle:  torch.Tensor, (n_triangles, n_points, n_dimensions), (m, 3, 3)
    :param epsilon: error tolerant
    :return: boolean (h_rays,)
    """

    E1 = triangle[:, 1] - triangle[:, 0]  # vector of edge 1 on triangle
    E2 = triangle[:, 2] - triangle[:, 0]  # vector of edge 2 on triangle
    N = torch.cross(E1, E2, dim=1)  # normal to E1 and E2

    invdet = 1. / -torch.einsum('ji, i -> j', ray_direction, N)  # inverse determinant

    A0 = ray_origins - triangle[0]
    # print('A0.shape: ', A0.shape)
    # print('ray_direction.shape: ', ray_direction.shape)
    DA0 = torch.cross(A0, ray_direction.repeat(A0.size(0), 1), dim=1)

    u = torch.einsum('ji, i -> j', DA0, E2) * invdet
    v = -torch.einsum('ji, i -> j', DA0, E1) * invdet
    t = torch.einsum('ji, i -> j', A0, N) * invdet

    intersection = (t >= 0.0) * (u >= 0.0) * (v >= 0.0) * ((u + v) <= 1.0)

    return intersection


def delaunay_mesh_in_2d(vertices):
    vertices2d = vertices[:, :2]
    tri = Delaunay(vertices2d)
    return vertices, tri.simplices


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def create_grid(N=256, mask_size=1., bbox_size=2.):
    N = int(N)
    grid_length = bbox_size / N
    s = np.arange(N)
    x, y, z = np.meshgrid(s, s, s)
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    grid_points_int = np.vstack((x, y, z)).T
    grid_points = grid_points_int.astype(float)
    grid_points -= np.ones(3) * N / 2.
    grid_points *= grid_length

    vector_len = np.linalg.norm(grid_points, axis=1)
    mask = np.where(vector_len <= mask_size, 1, 0).astype(np.int)
    # mask[grid_points[:, 0] > 0] = 0
    return grid_points, grid_points_int, mask.reshape((N, N, N)).astype(bool)


def create_grid_2d(N=256):
    N = int(N)
    grid_length = 1. / N
    s = np.arange(N)
    x, y = np.meshgrid(s, s)
    x, y = x.flatten(), y.flatten()
    grid_points_int = np.vstack((x, y)).T
    grid_points = grid_points_int.astype(float)
    grid_points -= np.ones(3) * N / 2.
    grid_points *= grid_length

    return grid_points, grid_points_int


def get_mgrid(side_len, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1."""
    if isinstance(side_len, int):
        side_len = dim * (side_len,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:side_len[0], :side_len[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (side_len[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (side_len[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:side_len[0], :side_len[1], :side_len[2]], axis=-1)[None, ...].astype(
            np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(side_len[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (side_len[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (side_len[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.tensor(pixel_coords).view(-1, dim)
    return pixel_coords
