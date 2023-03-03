import torch
import os
from utils import geometry
import numpy as np
from skimage.measure import marching_cubes
import traceback
import trimesh


def convert_sdf_samples_to_mesh(sdf_3d, mask, voxel_grid_origin=np.array([-1, -1, -1]), offset=None, scale=None,
                                level=0.0, return_value=False):
    """
    Convert sdf samples to .ply with color-coded template coordinates
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = np.array(sdf_3d)
    voxel_size = 2.0 / (numpy_3d_sdf_tensor.shape[0] - 1)

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = marching_cubes(numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3,
                                                       mask=mask)
    except:
        traceback.print_exc()
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    if return_value:
        return mesh_points, faces, values
    return mesh_points, faces


def export_mean(config, test_decoder, save_path, device, id, exp):
    test_decoder.eval()
    vox_resolution = int(config.voxel_resolution)
    grid, vox_idx, mask = geometry.create_grid(vox_resolution, mask_size=1.)
    grid_tensor = torch.tensor(grid)
    print(grid_tensor.shape)
    exp_mean = np.load(os.path.join(
        save_path, 'PCA', 'embeddings', 'exp_mean_embedding.npy'))
    id_mean = np.load(os.path.join(save_path, 'PCA',
                      'embeddings', 'id_mean_embedding.npy'))


    # >>>>>>>>>> Export Mean Face >>>>>>>>>>
    sdf_val = test_decoder.inference_by_batch(grid_tensor, torch.FloatTensor(exp_mean).to(device),
                                               torch.FloatTensor(
                                                   id_mean).to(device), device,
                                               config.points_per_inference, int_idx=False)
    print(sdf_val.shape)
    sdf_grid = sdf_val.reshape(
        (vox_resolution, vox_resolution, vox_resolution))
    points, faces = convert_sdf_samples_to_mesh(sdf_grid, mask)
    trimesh.Trimesh(points, faces).export(os.path.join(save_path, 'mean.obj'))
    test_decoder.train()

def export_eval(config, test_decoder, save_path, device, id, exp, epoch, residual):
    test_decoder.eval()
    residual.eval()
    vox_resolution = int(256)
    grid, vox_idx, mask = geometry.create_grid(vox_resolution, mask_size=1.)
    grid_tensor = torch.tensor(grid).to(device)

    z = []
    # >>>>>>>>>> Export Mean Face >>>>>>>>>>
    for i, pnts in enumerate(torch.split(grid_tensor, 100000, dim=0)):
        z.append(test_decoder(pnts.unsqueeze(0).to(torch.float32), exp, id).detach().cpu().numpy() +
                 residual(pnts.to(torch.float32)).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    sdf_grid = z.reshape(
        (vox_resolution, vox_resolution, vox_resolution))
    points, faces = convert_sdf_samples_to_mesh(sdf_grid, mask)
    x_axis = points[:, 1].copy()
    y_axis = points[:, 0].copy()
    points[:, 0] = x_axis
    points[:, 1] = y_axis
    x_axis = faces[:, 1].copy()
    y_axis = faces[:, 0].copy()
    faces[:, 0] = x_axis
    faces[:, 1] = y_axis
    trimesh.Trimesh(points, faces).export(os.path.join(save_path, '{0}.obj'.format(epoch)))
    test_decoder.train()
    residual.train()
