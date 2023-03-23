import torch
import torch.nn as nn
from render_model.film import *
from render_model.siren import *
from render_model.density import LaplaceDensity
import e3nn.o3 as o3


def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)


class RefNetwork(nn.Module):
    def __init__(
            self,
            d_in,
            d_out,
            dims,
            low_rank,
            brdf_network,
            calibration_network
    ):
        super().__init__()
        self.hidden_dim = dims[0]
        dims = list(dims)
        dims = [d_in] + dims + [d_out]
        self.skip = 4
        self.density = LaplaceDensity('cuda', params_init={'beta': 0.01}, beta_min=0.0001)
        self.num_layers = len(dims)

        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # spatial network
        self.num_rough = 1
        self.num_coeff = low_rank
        self.num_specular = 1
        self.num_diffuse = 3
        net_list = []
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            if l == 0:
                net_list.append(ModulatedLayer(dims[l], out_dim, first_layer_sine_init))
            elif l == self.skip:
                net_list.append(ModulatedLayer(dims[l] + dims[0], out_dim, sine_init))
            elif l < self.num_layers - 2:
                net_list.append(ModulatedLayer(dims[l], out_dim, sine_init))
            else:
                self.final_layer = torch.nn.Linear(256, self.num_rough + self.num_specular + self.num_diffuse + self.num_coeff)
                _xavier_init(self.final_layer)
        self.spatial_network = nn.ModuleList(net_list)

        # neural brdf network
        net_depth_condition = brdf_network.depth
        net_width_condition = brdf_network.hidden_dim
        layers = []
        for i in range(net_depth_condition):
            if i == 0:
                dim_in = 1 + 6 + 1
                dim_out = net_width_condition
            else:
                dim_in = net_width_condition
                dim_out = net_width_condition
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
        self.brdf_network = torch.nn.ModuleList(layers)
        self.basis_layer = torch.nn.Linear(net_width_condition, self.num_coeff)
        _xavier_init(self.basis_layer)

        # calibration network
        layers = []
        net_depth_condition = calibration_network.depth
        net_width_condition = calibration_network.hidden_dim
        for i in range(net_depth_condition):
            if i == 0:
                dim_in = 50
                dim_out = net_width_condition
            elif i == net_depth_condition - 1:
                dim_in = net_width_condition
                dim_out = net_width_condition
            else:
                dim_in = net_width_condition
                dim_out = net_width_condition
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
        linear = torch.nn.Linear(net_width_condition, 9)
        _xavier_init(linear)
        layers.append(linear)
        self.calibration_network = torch.nn.Sequential(*layers)

    def forward(self, points, view_dirs, normals=None, calibration_code=None, sh_light=None):
        rendering_input = torch.cat([points], dim=-1)
        x = rendering_input
        for index, layer in enumerate(self.spatial_network):
            if index % self.skip == 0 and index > 0:
                x = torch.cat([x, rendering_input], dim=-1)
            x = layer(x)

        y = self.final_layer(x)
        normals = normals.squeeze(0)

        roughness = self.softplus(y[..., 0:1])
        basis_coeff = y[..., 1:self.num_coeff + 1]
        albedo = self.sigmoid(y[..., self.num_coeff + 1:self.num_coeff + 4])
        spec = self.sigmoid(y[..., self.num_coeff + 4:])

        irradiance = e3_SH(lmax=10, directions=normals, sh=sh_light, lambert_coeff=True)
        irradiance = torch.relu(irradiance)
        diffuse = albedo * irradiance

        w0 = -view_dirs
        dot = normals * w0
        dot = dot.sum(dim=-1, keepdim=True)
        wr = 2 * normals * dot - w0
        light_transport = e3_SH(lmax=10, directions=wr, sh=sh_light, rho=roughness)
        light_transport = torch.relu(light_transport)
        # As roughness has an effect on BRDF basis, use roughness as input slightly improve the decompose result
        x = torch.cat([dot, w0, normals, roughness], dim=-1)
        for index, layer in enumerate(self.brdf_network):
            x = layer(x)
        brdf = self.basis_layer(x)
        cs = brdf * basis_coeff
        cs = cs.sum(dim=-1, keepdim=True)

        spec_energy = spec * light_transport * self.sigmoid(cs)
        diffuse_energy = diffuse
        raw_rgb = diffuse + spec * light_transport * self.sigmoid(cs)
        albedo_energy = albedo

        exposure = self.calibration_network(calibration_code)
        affine = exposure[:, :9].reshape(3, 3)

        raw_rgb = torch.matmul(raw_rgb, affine)
        raw_rgb = torch.clamp(raw_rgb, 0.0, 1.0)
        diff_ = torch.matmul(diffuse_energy, affine)
        diff_ = torch.clamp(diff_, 0.0, 1.0)
        spec_ = torch.matmul(spec_energy, affine)
        spec_ = torch.clamp(spec_, 0.0, 1.0)
        albedo = torch.matmul(albedo_energy * 3.1415926, affine)
        albedo = torch.clamp(albedo, 0.0, 1.0)

        light_reg = torch.cat([light_transport, irradiance], dim=0)
        output = {
            'raw_rgb': raw_rgb,
            'diffuse': diff_,
            'albedo': albedo,
            'spec': spec_,
            'spec_energy': spec_energy,
            'albedo_energy': albedo_energy,
            'diffuse_energy': diffuse_energy,
            'light_reg': light_reg,
        }

        return output


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


def e3_SH(lmax, directions, sh=None, rho=None, lambert_coeff=False):
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
