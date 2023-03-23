import sys, os

root_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(root_dir)
import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageEnhance
from render_model.imface_system_finetune import ImfaceSystem
from dataset.facescape_multi_dataset import FaceScape as TrainFacescape
from dataset.facescape_eval_dataset import FaceScape as EvalFacescape
import utils.general as utils
import utils.plots as plt
from glob import glob
from easydict import EasyDict as edict
import yaml
from utils import fileio, summary
from omegaconf import OmegaConf, DictConfig
from skimage.metrics import structural_similarity as ssim
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", help="Path to ckpt.")
parser.add_argument("--out_dir", help="Output directory.", type=str, required=True)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def fit(gt_img, recon_img, mask):
    '''
    gt_img,recon_img: (h,w,3)
    mask: (h,w)
    '''
    indices = np.where(mask==1)
    gt_img_pixels = gt_img[indices] # Nx3
    recon_img_pixels = recon_img[indices] # Nx3
    R = np.linalg.pinv(recon_img_pixels).dot(gt_img_pixels)
    return recon_img.dot(R), R

def load_config(path):
    with open(path, 'r') as f:
        config = f.read()
    config = edict(yaml.load(config, Loader=yaml.FullLoader))
    train_config = glob(os.path.join(config.load_path, '*.yaml'))
    assert len(train_config) == 1
    train_config = train_config[0]
    train_config = OmegaConf.load(train_config)
    train_config = DictConfig(train_config, flags={"allow_objects": True})
    id2idx = fileio.read_dict(os.path.join(config.load_path, 'id2idx.json'))
    exp2idx = fileio.read_dict(os.path.join(config.load_path, 'exp2idx.json'))
    return config, train_config, id2idx, exp2idx

def main(args):
    config_path = os.path.join(root_dir, 'config/train_neuface.yaml')
    config, train_config, id2idx, exp2idx = load_config(config_path)
    eval_type = config.eval_type
    id2idx = fileio.read_dict(os.path.join(config.load_path, 'id2idx.json'))
    exp2idx = fileio.read_dict(os.path.join(config.load_path, 'exp2idx.json'))
    train_config.DATA_CONFIG.ID_NUM = len(id2idx)
    train_config.DATA_CONFIG.EXP_NUM = len(exp2idx)
    train_config.DATA_CONFIG.TEMPLATE_KPTS = torch.zeros((68, 3)).float()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ImfaceSystem.load_from_checkpoint(args.ckpt, config=config, train_config=train_config, id2idx=id2idx,
                                              exp2idx=exp2idx).to(device).eval()
    config = model.config
    exp_name = config.expname
    if eval_type == 'trainset':
        test_dataset = TrainFacescape(config.dataset.data_dir, 4, split='train', facescape_id=config.dataset.facescape_id, facescape_exp=config.dataset.facescape_exp)
    elif eval_type == 'testset':
        # render novel view
        test_dataset = EvalFacescape(config.dataset.data_dir, 4)
    elif eval_type == 'evalset':
        test_dataset = TrainFacescape(config.dataset.data_dir, 4, split='test', facescape_id=config.dataset.facescape_id, facescape_exp=config.dataset.facescape_exp)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              collate_fn=test_dataset.collate_fn
                                              )

    save_path = os.path.join(args.out_dir, 'test', exp_name)
    os.makedirs(save_path, exist_ok=True)

    id_init = 0
    exp_init = 0
    id_latent = model.id_lat_vecs(torch.tensor(id_init).to(device)).unsqueeze(0)
    exp_latent = model.exp_lat_vecs(torch.tensor(exp_init).to(device)).unsqueeze(0)
    summary.export_eval(config, model.decoder, save_path, device, id_latent, exp_latent,
                        '{0}_{1}'.format(id_init, exp_init), model.residual)
    model.eval()
    all_psnr = []
    all_ssim = []
    for data_index, (indices, model_input, ground_truth) in enumerate(tqdm(test_loader)):
        model_input["intrinsics"] = model_input["intrinsics"].to(device)
        model_input["uv"] = model_input["uv"].to(device)
        model_input["object_mask"] = model_input["object_mask"].to(device)
        out_num = test_dataset.exp_num[indices]
        id_latent = model.id_lat_vecs(torch.tensor(0).to(device)).unsqueeze(0)
        exp_latent = model.exp_lat_vecs(torch.tensor(0).to(device)).unsqueeze(0)
        if eval_type == 'trainset':
            calibration_code = model.calibration_lat_vecs(torch.tensor(indices).to(device))
            model_input['calibration_code'] = calibration_code
        else:
            calibration_code = model.calibration_lat_vecs(torch.tensor([6]).to(device))
            model_input['calibration_code'] = calibration_code
        L = model.sh_light(torch.tensor([0]).to(calibration_code.device))
        L = L.reshape(-1, 121, 3)
        model_input['sh_light'] = L
        model_input['id_latent'] = id_latent
        model_input['exp_latent'] = exp_latent
        model_input['pose'] = model_input['pose'].to(device)
        img_res = [test_dataset.h[indices], test_dataset.w[indices]]
        total_pixels = img_res[0] * img_res[1]
        split = utils.split_input(model_input, total_pixels)
        res = []
        for s in split:
            out = model(s)
            res.append({
                'points': out['points'].detach(),
                'rgb_values': out['rgb_values'].detach(),
                'diffuse_values': out['diffuse_values'].detach(),
                'albedo_values': out['albedo_values'].detach(),
                'spec_values': out['spec_values'].detach(),
                'network_object_mask': out['network_object_mask'].detach(),
                'object_mask': out['object_mask'].detach(),
                'normals': out['normals'].detach(),
            })

        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, total_pixels, batch_size)
        if eval_type == 'trainset' or eval_type == 'evalset':
            mask = model_outputs['object_mask']
        elif eval_type == 'testset':
            mask = model_outputs['network_object_mask']
        mask = mask.reshape(batch_size, total_pixels, 1)
        mask = plt.lin2img(mask, img_res)
        mask = mask.squeeze(0).cpu().detach().numpy()
        mask = mask.transpose(1, 2, 0).astype(np.uint8)
        rgb_eval = model_outputs['rgb_values']
        rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
        rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
        rgb_eval = rgb_eval.transpose(1, 2, 0)
        rgb_pred = rgb_eval
        rgb_eval = np.clip(rgb_eval, 0, 1)
        rgb_eval = (rgb_eval * 255).astype(np.uint8)
        rgb_eval[mask.squeeze(-1) == 0] = 255
        img = Image.fromarray(rgb_eval)
        if eval_type == 'testset':
            img = img.rotate(90)
        img.save(os.path.join(save_path, '{1}_{0}.png'.format('%03d' % out_num, 1)))

        if eval_type == 'trainset' or eval_type == 'evalset':
            rgb_eval = ground_truth['rgb']
            rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
            rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
            rgb_eval = rgb_eval.transpose(1, 2, 0)
            rgb_gt = rgb_eval
            # Eliminate the effect of exposure by applying a calibration matrix
            rgb_pred, R = fit(rgb_gt, rgb_pred, mask.squeeze(-1))
            rgb_pred = np.clip(rgb_pred, 0, 1)
            rgb_eval = (rgb_pred * 255).astype(np.uint8)
            rgb_eval[mask.squeeze(-1) == 0] = 0
            img = Image.fromarray(rgb_eval)
            img.save(os.path.join(save_path, '{1}_{0}_fit.png'.format('%03d' % out_num, 1)))
            rgb_gt = rgb_gt * mask
            rgb_pred = rgb_pred * mask
            rgb_gt2 = rgb_gt[mask.squeeze(-1) == 1]
            rgb_eval2 = rgb_pred[mask.squeeze(-1) == 1]
            ss = ssim(rgb_eval2, rgb_gt2, multichannel=True)
            all_ssim.append(ss)
            ps = calc_psnr(torch.tensor(rgb_pred), torch.tensor(rgb_gt), torch.tensor(mask))
            all_psnr.append(ps)

        rgb_eval = model_outputs['diffuse_values']
        rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
        rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
        rgb_eval = rgb_eval.transpose(1, 2, 0)
        rgb_eval = (rgb_eval * 255).astype(np.uint8)
        rgb_eval[mask.squeeze(-1) == 0] = 255
        img = Image.fromarray(rgb_eval)
        if eval_type == 'testset':
            img = img.rotate(90)
        img.save(os.path.join(save_path, '{1}_{0}.png'.format('%03d_diffuse' % out_num, 1)))

        rgb_eval = model_outputs['albedo_values']
        rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
        rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
        rgb_eval = rgb_eval.transpose(1, 2, 0)
        rgb_eval = (rgb_eval * 255).astype(np.uint8)
        rgb_eval[mask.squeeze(-1) == 0] = 255
        img = Image.fromarray(rgb_eval)
        if eval_type == 'testset':
            img = img.rotate(90)
        img.save(os.path.join(save_path, '{1}_{0}.png'.format('%03d_albedo' % out_num, 1)))

        rgb_eval = model_outputs['spec_values']
        rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
        rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
        rgb_eval = rgb_eval.transpose(1, 2, 0)
        rgb_eval = (rgb_eval * 255).astype(np.uint8)
        img = Image.fromarray(rgb_eval)
        img = ImageEnhance.Brightness(img)
        img = img.enhance(2.5)
        img = img.convert('L')
        img = np.stack((np.array(img),) * 3, axis=-1)
        img[mask.squeeze(-1) == 0] = 255
        img = Image.fromarray(img)
        if eval_type == 'testset':
            img = img.rotate(90)
        img.save(os.path.join(save_path, '{1}_{0}.png'.format('%03d_spec' % out_num, 1)))

        normal = model_outputs['normals']
        normal = normal.reshape(batch_size, total_pixels, 3)
        l1_normal = torch.norm(normal, dim=2)
        normal /= l1_normal.unsqueeze(-1)
        normal = plt.lin2img(normal, img_res)
        normal = normal * 127.5 + 127.5
        normal = normal.squeeze(0).cpu().detach().numpy()
        normal = normal.transpose(1, 2, 0).astype(np.uint8)
        normal[mask.squeeze(-1) == 0] = 255
        img = Image.fromarray(normal)
        if eval_type == 'testset':
            img = img.rotate(90)
        img.save(os.path.join(save_path, '{1}_{0}.png'.format('%03d_normal' % out_num, 1)))
    if eval_type == 'testset':
        print('render novel view succeed!')
    else:
        print('The mean psnr is: ', torch.mean(torch.tensor(all_psnr)).item())
        print('The mean ssim is: ', torch.mean(torch.tensor(all_ssim)).item())


def calc_psnr(x: torch.Tensor, y: torch.Tensor, mask):
    """
    Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
    """
    mse = torch.mean((x - y) ** 2) * x.shape[0] * x.shape[1] / mask.sum()
    psnr = -10.0 * torch.log10(mse)
    return psnr


if __name__ == '__main__':
    args = parser.parse_args()
    blender_scenes = main(args)
