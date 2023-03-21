import json
import os
import torch
import numpy as np
import trimesh
import utils.general as utils
from utils import rend_util
from tqdm import tqdm
from kornia import create_meshgrid

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w

def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

class FaceScape(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 data_dir, factor
                 ):

        self.instance_dir = os.path.join(data_dir)

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None

        self.intrinsics_all = []
        self.pose_all = []
        self.rgb_images = []
        self.h = []
        self.w = []
        self.object_masks = []
        self.id = []
        self.exp = []
        factor = factor
        h = 512
        w = 512
        focal = 1200
        K = np.array([
            [focal, 0, 0.5 * w],
            [0, focal, 0.5 * h],
            [0, 0, 1]
        ])
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K

        list_theta = [90]
        for th in tqdm(list_theta):
            for ph in range(20):
                c2w = pose_spherical(th, -48 + -ph * 4.2, 7.52455745)
                self.pose_all.append(c2w)
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
        id_idx = 1
        exp_idx = 1
        self.rgb_images = []
        self.h = []
        self.w = []
        self.exp_num = []
        for i in range(20):
            self.exp_num.append(i)
            self.h.append(h)
            self.w.append(w)
            rgb = np.zeros((3, 512 * 512))
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.id.append(torch.from_numpy(np.array(id_idx - 1)))
            self.exp.append(torch.from_numpy(np.array(exp_idx - 1)))
            self.rgb_images.append(torch.from_numpy(rgb).float())

        self.object_masks = []
        for i in range(20):
            object_mask = np.zeros((512 * 512))
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(object_mask).bool())
        self.n_images = len(self.pose_all)
        print('image num: {0}'.format(self.n_images))

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.h[idx], 0:self.w[idx]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        # total = self.h[idx] * self.w[idx]
        id_latent = self.id[idx]
        exp_latent = self.exp[idx]
        sample = {
            "object_mask": self.object_masks[idx],
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "id": id_latent,
            "exp": exp_latent
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        if self.sampling_idx is not None:
            s_idx = self.sampling_idx[idx]
            ground_truth["rgb"] = self.rgb_images[idx][s_idx, :]
            sample["object_mask"] = self.object_masks[idx][s_idx]
            sample["uv"] = uv[s_idx, :]

        sample["pose"] = self.pose_all[idx]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        # if sampling_size == -1:
        #     self.sampling_idx = None
        # else:
        #     self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = []
            for i in range(self.n_images):
                total = self.h[i] * self.w[i]
                s_idx = torch.randperm(total)[:sampling_size]
                self.sampling_idx.append(s_idx)