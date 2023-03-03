import torch, math
from torch import nn
import torch.nn.functional as F
import numpy as np


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
            nn.init.zeros_(m.bias)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper Sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
            nn.init.zeros_(m.bias)


def last_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)


class ModulatedLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, init_func):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        if init_func is not None:
            self.layer.apply(init_func)

    def forward(self, x, phase_shift=None, freq=30):
        x = self.layer(x)
        if phase_shift is None:
            return torch.sin(freq * x)
        else:
            phase_shift = phase_shift.unsqueeze(1).expand_as(x)
            return torch.sin(freq * (x + phase_shift))


class ModulatedSIREN(nn.Module):
    def __init__(self, code_dim, in_dim, out_dim, hidden_dim=32, map_hidden_dim=256, num_hidden_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        net_list = [ModulatedLayer(in_dim, hidden_dim, first_layer_sine_init)]
        for i in range(num_hidden_layers - 1):
            net_list.append(ModulatedLayer(hidden_dim, hidden_dim, sine_init))
        self.network = nn.ModuleList(net_list)
        self.out_layer = nn.Linear(hidden_dim, out_dim)