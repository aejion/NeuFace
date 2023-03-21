import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils import rend_util
from model.embedder import *
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork
from model.siren import *


# Basic SIREN fully connected layer
class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, std_init=1, freq_init=False, is_first=False):
        super().__init__()
        if is_first:
            self.weight = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-1 / in_dim, 1 / in_dim))
        elif freq_init:
            self.weight = nn.Parameter(
                torch.empty(out_dim, in_dim).uniform_(-np.sqrt(6 / in_dim) / 25, np.sqrt(6 / in_dim) / 25))
        else:
            self.weight = nn.Parameter(
                0.25 * nn.init.kaiming_normal_(torch.randn(out_dim, in_dim), a=0.2, mode='fan_in',
                                               nonlinearity='leaky_relu'))

        self.bias = nn.Parameter(nn.init.uniform_(torch.empty(out_dim), a=-np.sqrt(1 / in_dim), b=np.sqrt(1 / in_dim)))

        self.bias_init = bias_init
        self.std_init = std_init

    def forward(self, input):
        out = self.std_init * F.linear(input, self.weight, bias=self.bias) + self.bias_init

        return out


class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()
        dims = [d_in] + dims + [d_out + 1 + 9 + 3 + 3]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)


from model.siren import *


class BrdfNetwork(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()
        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.tanh = nn.Tanh()

        num_feature = 9
        self.num_feature = num_feature
        net_depth_condition = 4
        net_width_condition = 512
        layers = []
        for i in range(net_depth_condition):
            if i == 0:
                dim_in = 1 + 1 + 6
                dim_out = net_width_condition
            else:
                dim_in = net_width_condition
                dim_out = net_width_condition
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
        self.view_layers = torch.nn.ModuleList(layers)
        self.color_layer = torch.nn.Linear(net_width_condition, 27)
        _xavier_init(self.color_layer)

    def forward(self, normals, view_dirs, feature, ref_SH):
        roughness = self.softplus(feature[..., 0:1])
        coe_feature = feature[..., 1:self.num_feature + 1]
        albedo = self.sigmoid(feature[..., self.num_feature + 1:self.num_feature + 4])
        spec = self.sigmoid(feature[..., self.num_feature + 4:])

        irradiance = e3_SH(lmax=10, directions=normals, sh=ref_SH, lambert_coeff=True)
        irradiance = torch.relu(irradiance)
        diffuse = albedo * irradiance

        w0 = view_dirs
        dot = normals * w0
        dot = dot.sum(dim=-1, keepdim=True)
        wr = 2 * normals * dot - w0
        light_feature = e3_SH(lmax=10, directions=wr, sh=ref_SH, rho=roughness)
        light_feature = torch.relu(light_feature)
        x = torch.cat([dot, w0, normals, roughness], dim=-1)
        for index, layer in enumerate(self.view_layers):
            x = layer(x)
        cs = self.color_layer(x)
        r = cs[..., :9]
        r = torch.sum(r * coe_feature, dim=-1, keepdim=True)
        g = cs[..., 9:18]
        g = torch.sum(g * coe_feature, dim=-1, keepdim=True)
        b = cs[..., 18:]
        b = torch.sum(b * coe_feature, dim=-1, keepdim=True)
        cs = torch.cat([r, g, b], dim=-1)

        raw_rgb = diffuse + spec * light_feature * self.sigmoid(cs)
        sh_light = torch.cat([light_feature, irradiance], dim=0)

        return raw_rgb, albedo, diffuse, spec * light_feature * self.sigmoid(cs), sh_light


lambert_sh_k = [3.1415926535897927,
                2.0943951023931957,
                0.7853981633974483,
                0.0,
                -0.13089969389957473,
                0.0,
                0.04908738521234052,
                0.0,
                -0.024543692606170262,
                0.0,
                0.014317154020265985]

import e3nn.o3 as o3


def e3_SH(lmax, directions, sh, rho=None, lambert_coeff=False):
    d = directions[..., [1, 2, 0]]
    basis = torch.zeros((directions.shape[0], (lmax + 1) ** 2)).to(directions.device).float()
    for i in range(lmax + 1):
        if lambert_coeff == True:
            basis[..., i ** 2: (i + 1) ** 2] = o3.spherical_harmonics(i, d, normalize=False) * lambert_sh_k[i]
        else:
            basis[..., i ** 2: (i + 1) ** 2] = o3.spherical_harmonics(i, d, normalize=False) * torch.exp(
                - i * (i + 1) / 2 * rho)
    basis = basis.unsqueeze(-1)
    tmp = basis * sh  # (N, 25 or 9, 3)
    return tmp.sum(-2)  # (N, 3)


class calibrationMLP(nn.Module):
    def __init__(
            self
    ):
        super().__init__()
        layers = []
        layer_len = 1
        net_width_condition = 128
        for i in range(layer_len):
            if i == 0:
                dim_in = 50
                dim_out = net_width_condition
            elif i == layer_len - 1:
                dim_in = net_width_condition
                dim_out = 128
            else:
                dim_in = net_width_condition
                dim_out = net_width_condition
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
        linear = torch.nn.Linear(128, 9)
        _xavier_init(linear)
        layers.append(linear)
        self.exp_layer = torch.nn.Sequential(*layers)

    def forward(self, tex_latent):
        exposure = self.exp_layer(tex_latent)
        affine = exposure[:, :9].reshape(3, 3)
        return affine


class IDRNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        self.rendering_network = BrdfNetwork()
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')

        self.calibration = calibrationMLP()
        latent_size = 50
        self.tex_lat_vecs = torch.nn.Embedding(100, latent_size, max_norm=1.0)
        torch.nn.init.normal_(
            self.tex_lat_vecs.weight.data,
            0.0,
            0.1,
        )

        default_l = torch.tensor([[2.9861e+00, 3.4646e+00, 3.9559e+00],
                                  [1.0013e-01, -6.7589e-02, -3.1161e-01],
                                  [-8.2520e-01, -5.2738e-01, -9.7385e-02],
                                  [2.2311e-03, 4.3553e-03, 4.9501e-03],
                                  [-6.4355e-03, 9.7476e-03, -2.3863e-02],
                                  [1.1078e-01, -6.0607e-02, -1.9541e-01],
                                  [7.9123e-01, 7.6916e-01, 5.6288e-01],
                                  [6.5793e-02, 4.3270e-02, -1.7002e-01],
                                  [-7.2674e-02, 4.5177e-02, 2.2858e-01]]).float()
        default_l = default_l.reshape(1, -1)
        high_l = torch.zeros(1, 112 * 3).float()
        default_l = torch.cat([default_l, high_l], dim=-1)
        self.ref_SH = torch.nn.Parameter(default_l, requires_grad=True).reshape(1, 121, 3).cuda()

    def forward(self, input):
        # Parse model input
        tex_latent = input['tex_latent']
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs)
        self.implicit_network.train()

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

        sdf_output = self.implicit_network(points)[:, 0:1]
        ray_dirs = ray_dirs.reshape(-1, 3)

        g = self.implicit_network.gradient(points)

        s_normals = g[:, 0, :]
        s_normals = s_normals.reshape(-1, 3)

        if self.training:
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            points_all = torch.cat([surface_points, eikonal_points], dim=0)

            output = self.implicit_network(surface_points)
            surface_sdf_values = output[:N, 0:1].detach()

            g = self.implicit_network.gradient(points_all)
            surface_points_grad = g[:N, 0, :].clone().detach()
            grad_theta = g[N:, 0, :]

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad,
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None

        view = -ray_dirs[surface_mask]

        rgb_values = torch.zeros_like(points).float().cuda()
        sh_light = None
        spec_energy = None
        albedo_values = torch.zeros_like(points).float().cuda()
        diffuse_values = torch.zeros_like(points).float().cuda()
        spec_values = torch.zeros_like(points).float().cuda()
        if differentiable_surface_points.shape[0] > 0:
            affine = self.calibration(tex_latent)
            rgb_value, albedo, diffuse, spec, sh_light = self.get_rbg_value(differentiable_surface_points, view)

            rgb_value = torch.matmul(rgb_value, affine)
            rgb_value = torch.clamp(rgb_value, 0.0, 1.0)
            rgb_values[surface_mask] = rgb_value

            albedo = torch.matmul(albedo, affine)
            albedo = torch.clamp(albedo * 3.1415926, 0.0, 1.0)
            albedo_values[surface_mask] = albedo

            diffuse = torch.matmul(diffuse, affine)
            diffuse = torch.clamp(diffuse, 0.0, 1.0)
            diffuse_values[surface_mask] = diffuse

            spec = torch.matmul(spec, affine)
            spec = torch.clamp(spec, 0.0, 1.0)
            spec_values[surface_mask] = spec

        output = {
            'points': points,
            'rgb_values': rgb_values,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta,
            'normals': s_normals,
            'sh_light': sh_light,
            'spec_energy': spec_energy,

            'albedo_values': albedo_values,
            'diffuse_values': diffuse_values,
            'spec_values': spec_values,

        }

        return output

    def get_rbg_value(self, points, view_dirs):
        output = self.implicit_network(points)
        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]
        normals = torch.nn.functional.normalize(normals, dim=-1)
        feature = output[:, 1:]
        rgb_vals, albedo, diffuse, spec, sh_light = self.rendering_network(normals, view_dirs, feature, self.ref_SH)

        return rgb_vals, albedo, diffuse, spec, sh_light
