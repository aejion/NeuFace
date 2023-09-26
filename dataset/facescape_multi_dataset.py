import sys
sys.path.append('/Users/zhy/Desktop/neural_field_code/imface_render/')
import json
import os
import torch
import numpy as np
import trimesh
# import rend_util
# from utils import rend_util
from tqdm import tqdm
from kornia import create_meshgrid
import cv2
from PIL import Image
import skimage

def load_rgb_resize(path, factor):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H = img.shape[0]
    W = img.shape[1]

    img = cv2.resize(img, (W // factor, H // factor), interpolation=cv2.INTER_AREA)
    img = skimage.img_as_float32(img)
    # pixel values between [-1,1]
    # img -= 0.5
    # img *= 2.
    img = img.transpose(2, 0, 1)
    return img

def load_mask_resize(path, factor):
    alpha = cv2.imread(path, 0)
    H = alpha.shape[0]
    W = alpha.shape[1]
    alpha = cv2.resize(alpha, (W // factor, H // factor), interpolation=cv2.INTER_AREA)
    alpha = skimage.img_as_float32(alpha)
    object_mask = alpha > 0.5

    return object_mask

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

def calc_psnr(x: torch.Tensor, y: torch.Tensor, mask):
    """
    Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
    """
    mse = torch.mean((x - y) ** 2) * x.shape[0] * x.shape[1] / mask.sum()
    psnr = -10.0 * torch.log10(mse)
    return psnr

class FaceScape(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 data_dir, factor, split, facescape_id, facescape_exp):
        self.instance_dir = os.path.join(data_dir)
        print(self.instance_dir)
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None

        image_dir = '{0}/imface_image'.format(self.instance_dir)  # path to image
        mask_dir = '{0}/imface_mask'.format(self.instance_dir)    # path to mask
        params_dir = '{0}/imface_image'.format(self.instance_dir) # path to camera parameters
        ply_path = '{0}/imface_model'.format(self.instance_dir)   # path to the TU model
        expression = ['neutral', 'smile', 'mouth_stretch', 'anger', 'jaw_left', 'jaw_right', 'jaw_forward',
                      'mouth_left', 'mouth_right', 'dimpler', 'chin_raiser',
                      'lip_puckerer', 'lip_funneler', 'sadness', 'lip_roll', 'grin', 'cheek_blowing', 'eye_closed',
                      'brow_raiser', 'brow_lower']
        self.intrinsics_all = []
        self.pose_all = []
        self.rgb_images = []
        self.h = []
        self.w = []
        self.object_masks = []
        self.id = []
        self.exp = []
        self.exp_num = []
        self.sample_list = []
        self.have_sample = []

        with open(os.path.join(self.instance_dir, f"Rt_scale_dict.json"), 'r') as f:
            Rt_scale_dict = json.load(f)

        land_dir = os.path.join(self.instance_dir, 'landmark_indices.npz')
        landmark = np.load(land_dir)['v10'][30]
        factor = factor

        i = facescape_id
        j = facescape_exp
        i_dir = os.path.join(image_dir, str(i), '{0}_{1}'.format(j, expression[j-1]), split)
        par_dir = os.path.join(params_dir, str(i), '{0}_{1}'.format(j, expression[j-1]))
        with open(os.path.join(par_dir, f"params.json"), 'r') as f:
            meta = json.load(f)
        image_path = os.listdir(i_dir)
        image_path.sort()
        scale = Rt_scale_dict['%d' % i]['%d' % j][0]
        Rt = np.array(Rt_scale_dict['%d' % i]['%d' % j][1])


        ply_name = '{0}_{1}.obj'.format(j, expression[j - 1])
        ply = os.path.join(ply_path, str(i), ply_name)
        mview_mesh = trimesh.load(ply, process=False, maintain_order=True)
        # align multi-view model to TU model
        point = mview_mesh.vertices[landmark]

        for p in image_path:
            if not p.endswith('.jpg'): continue
            self.id.append(torch.tensor(0))
            self.exp.append(torch.tensor(0))
            num = p[:-4]
            self.exp_num.append(int(num))
            r_path = os.path.join(i_dir, p)
            rgb = load_rgb_resize(r_path, factor)

            self.h.append(rgb.shape[1])
            self.w.append(rgb.shape[2])
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

            m_path = os.path.join(mask_dir, str(i), '{0}_{1}'.format(j, expression[j-1]), split, num + '.jpg')
            object_mask = load_mask_resize(m_path, factor)
            h, w = object_mask.shape
            x_axis, y_axis = np.where(object_mask == True)
            x_min = max(x_axis.min() - 30, 0)
            x_max = min(x_axis.max() + 30, h - 1)
            y_min = max(y_axis.min() - 30, 0)
            y_max = min(y_axis.max() + 30, w - 1)
            sam_list = []
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    sam_list.append(x * w + y)
            # train with the pixels in the mask area
            self.sample_list.append(torch.tensor(sam_list))
            self.have_sample.append(torch.zeros(h * w) + 255)
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(object_mask).bool())

            # obtain camera's intrinsics and extrinsics
            K = np.array(meta[num + '_K'])
            K[:2, :3] /= factor
            pose = np.array(meta[num + '_Rt'])[:3, :4]
            R_cv2gl = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            pose = R_cv2gl.dot(pose)
            cam_pose = np.eye(4)
            cam_pose[:3, :3] = pose[:3, :3].T
            cam_pose[:3, 3] = -pose[:3, :3].T.dot(pose[:, 3])

            rays_o = cam_pose[:3, 3]
            rays_o *= scale
            rays_o = np.dot(Rt[:3, :3], rays_o.T).T + Rt[:3, 3]
            rays_o -= point
            rays_o[2] += 40
            rays_o /= 100

            r = cam_pose[:3, :3]
            r = np.dot(Rt[:3, :3], r)

            cam_pose[:3, 3] = rays_o
            cam_pose[:3, :3] = r
            intrinsics = np.eye(4)
            intrinsics[:3, :3] = K

            self.pose_all.append(torch.from_numpy(cam_pose).float())
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
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
            "exp": exp_latent,
            # "key_pts": self.key_pts[idx]
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
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = []
            for i in range(self.n_images):
                total = self.sample_list[i].shape[0]
                s_idx = torch.randperm(total)[:sampling_size]
                self.have_sample[i][self.sample_list[i][s_idx]] = 0
                self.sampling_idx.append(self.sample_list[i][s_idx])