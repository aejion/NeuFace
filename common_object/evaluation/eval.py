import sys
sys.path.append('../common_object')
import argparse
import GPUtil
import os
from pyhocon import ConfigFactory
import torch
import numpy as np
import cvxpy as cp
from PIL import Image
import math
from skimage.metrics import structural_similarity as ssim
import e3nn.o3 as o3
import utils.general as utils
import utils.plots as plt
from utils import rend_util
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


def e3_SH(lmax, directions, sh, rho):
    d = directions[..., [1, 2, 0]]
    basis = torch.zeros((directions.shape[0], (lmax + 1) ** 2)).to(directions.device).float()
    for i in range(lmax + 1):
        basis[..., i ** 2: (i + 1) ** 2] = o3.spherical_harmonics(i, d, normalize=False) * torch.exp(
            - i * (i + 1) / 2 * rho)
    basis = basis.unsqueeze(-1)
    tmp = basis * sh  # (N, 25 or 9, 3)
    return tmp.sum(-2).reshape(1, 100, 200, 3).numpy()  # (N, 3)


def evaluate(**kwargs):
    torch.set_default_dtype(torch.float32)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']
    eval_cameras = kwargs['eval_cameras']
    eval_rendering = kwargs['eval_rendering']

    expname = conf.get_string('train.expname') + kwargs['expname']
    scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else conf.get_int('dataset.scan_id', default=-1)
    if scan_id != -1:
        expname = expname + '_{0}'.format(scan_id)

    if kwargs['timestamp'] == 'latest':
        if os.path.exists(os.path.join(kwargs['exps_folder_name'], expname)):
            timestamps = os.listdir(os.path.join(kwargs['exps_folder_name'], expname))
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER')
                exit()
            else:
                timestamp = sorted(timestamps)[-1]
        else:
            print('WRONG EXP FOLDER')
            exit()
    else:
        timestamp = kwargs['timestamp']

    utils.mkdir_ifnotexists(os.path.join(evals_folder_name))
    expdir = os.path.join(exps_folder_name, expname)
    evaldir = os.path.join(evals_folder_name, expname)
    utils.mkdir_ifnotexists(evaldir)

    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'))
    if torch.cuda.is_available():
        model.cuda()

    dataset_conf = conf.get_config('dataset')
    if kwargs['scan_id'] != -1:
        dataset_conf['scan_id'] = kwargs['scan_id']
    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(eval_cameras, **dataset_conf)

    if eval_rendering:
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      collate_fn=eval_dataset.collate_fn
                                                      )

    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')
    print(old_checkpnts_dir)
    saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
    model.load_state_dict(saved_model_state["model_state_dict"])
    epoch = saved_model_state['epoch']

    ####################################################################################################################
    print("evaluating...")

    model.eval()

    with torch.no_grad():

        mesh = plt.get_surface_high_res_mesh(
            sdf=lambda x: model.implicit_network(x)[:, 0],
            resolution=kwargs['resolution']
        )

        components = mesh.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float)
        mesh_clean = components[areas.argmax()]
        mesh_clean.export('{0}/surface_world_coordinates_{1}.ply'.format(evaldir, epoch), 'ply')

    if eval_rendering:
        images_dir = '{0}/rendering'.format(evaldir)
        utils.mkdir_ifnotexists(images_dir)
        psnrs = []
        ssims = []
        for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):
            tex_latent = model.tex_lat_vecs(torch.tensor([4]).cuda())
            model_input['tex_latent'] = tex_latent
            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input["object_mask"] = model_input["object_mask"].cuda()
            model_input['pose'] = model_input['pose'].cuda()
            img_res = [eval_dataset.h[data_index], eval_dataset.w[data_index]]
            total_pixels = img_res[0] * img_res[1]
            split = utils.split_input(model_input, total_pixels)
            res = []
            for s in split:
                out = model(s)
                res.append({
                    'rgb_values': out['rgb_values'].detach(),
                    'albedo_values': out['albedo_values'].detach(),
                    'diffuse_values': out['diffuse_values'].detach(),
                    'spec_values': out['spec_values'].detach(),
                    'normals': out['normals'].detach(),
                    'network_object_mask': out['network_object_mask'].detach(),
                })

            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, total_pixels, batch_size)

            mask = model_outputs["network_object_mask"]
            mask = mask.reshape(batch_size, total_pixels, 1)
            mask = plt.lin2img(mask, img_res)
            mask = mask.squeeze(0).cpu().detach().numpy()
            mask = mask.transpose(1, 2, 0).astype(np.uint8)

            rgb_gt = ground_truth['rgb']
            rgb_gt = plt.lin2img(rgb_gt, img_res).numpy()[0]
            rgb_gt = rgb_gt.transpose(1, 2, 0)

            rgb_eval = model_outputs['rgb_values']
            rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
            rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
            rgb_eval = rgb_eval.transpose(1, 2, 0)
            rgb_pred = rgb_eval
            rgb_eval[mask.squeeze() == 0] = 1
            img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
            img.save('{0}/rgb_{1}.png'.format(images_dir, '%03d' % indices[0]))

            rgb_pred, R = fit(rgb_gt, rgb_pred, mask.squeeze(-1))
            rgb_pred = np.clip(rgb_pred, 0, 1)
            rgb_eval2 = (rgb_pred * 255).astype(np.uint8)
            rgb_eval2[mask.squeeze(-1) == 0] = 255
            img = Image.fromarray(rgb_eval2)
            img.save('{0}/eval_{1}_fit.png'.format(images_dir, '%03d' % indices[0]))


            rgb_eval = model_outputs['albedo_values']
            rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
            rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
            rgb_eval = rgb_eval.transpose(1, 2, 0)
            rgb_eval = np.clip(rgb_eval, 0, 1)
            rgb_eval[mask.squeeze() == 0] = 1
            img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
            img.save('{0}/albedo_{1}.png'.format(images_dir, '%03d' % indices[0]))

            rgb_eval = model_outputs['diffuse_values']
            rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
            rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
            rgb_eval = rgb_eval.transpose(1, 2, 0)
            rgb_eval = np.clip(rgb_eval, 0, 1)
            rgb_eval[mask.squeeze() == 0] = 1
            img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
            img.save('{0}/diffuse_{1}.png'.format(images_dir, '%03d' % indices[0]))

            rgb_eval = model_outputs['spec_values']
            rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
            rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
            rgb_eval = rgb_eval.transpose(1, 2, 0)
            rgb_eval = np.clip(rgb_eval, 0, 1)
            rgb_eval[mask.squeeze() == 0] = 1
            img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
            img.save('{0}/spec_{1}.png'.format(images_dir, '%03d' % indices[0]))

            normal = model_outputs['normals']
            normal = normal.reshape(batch_size, total_pixels, 3)
            l1_normal = torch.norm(normal, dim=2)
            normal /= l1_normal.unsqueeze(-1)
            normal = plt.lin2img(normal, img_res)
            normal = normal * 127.5 + 127.5
            normal = normal.squeeze(0).cpu().detach().numpy()
            normal = normal.transpose(1, 2, 0).astype(np.uint8)
            normal = normal * mask
            normal[mask.squeeze() == 0] = 255
            img = Image.fromarray(normal)
            img.save('{0}/normal_{1}.png'.format(images_dir,'%03d' % indices[0]))

            rgb_eval = rgb_pred * mask
            rgb_gt = rgb_gt * mask
            rgb_gt2 = rgb_gt[mask.squeeze(-1) == 1]
            rgb_eval2 = rgb_eval[mask.squeeze(-1) == 1]
            ss = ssim(rgb_eval2, rgb_gt2, multichannel=True)
            ssims.append(ss)
            psnr = calculate_psnr(rgb_eval, rgb_gt, mask)
            psnrs.append(psnr)
        psnrs = np.array(psnrs).astype(np.float64)
        ssims = np.array(ssims).astype(np.float64)
        print("RENDERING EVALUATION {2}: psnr mean = {0} ; psnr std = {1}".format("%.2f" % psnrs.mean(), "%.2f" % psnrs.std(), scan_id))
        print('ssim mean = {0} ; ssim std = {1}'.format(ssims.mean(), ssims.std()))

def compare_rotations(R1, R2):
    cos_err = (torch.bmm(R1, R2.transpose(1, 2))[:, torch.arange(3), torch.arange(3)].sum(dim=-1) - 1) / 2
    cos_err[cos_err > 1] = 1
    cos_err[cos_err < -1] = -1
    return cos_err.acos() * 180 / np.pi

def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2) * (img2.shape[0] * img2.shape[1]) / mask.sum()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/dtu_fixed_cameras.conf')
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--timestamp', default='latest', type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--resolution', default=256, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--is_uniform_grid', default=False, action="store_true", help='If set, evaluate marching cube with uniform grid.')
    parser.add_argument('--eval_cameras', default=False, action="store_true", help='If set, evaluate camera accuracy of trained cameras.')
    parser.add_argument('--eval_rendering', default=False, action="store_true", help='If set, evaluate rendering quality.')

    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    evaluate(conf=opt.conf,
             expname=opt.expname,
             exps_folder_name=opt.exps_folder,
             evals_folder_name='evals',
             timestamp=opt.timestamp,
             checkpoint=opt.checkpoint,
             scan_id=opt.scan_id,
             resolution=opt.resolution,
             eval_cameras=opt.eval_cameras,
             eval_rendering=opt.eval_rendering
             )
