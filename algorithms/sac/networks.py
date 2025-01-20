import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sac import utils

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class NoAug():
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return x


class ParallelizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int,
                bias: bool=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ParallelizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.empty((ensemble_size, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        return input @ self.weight + self.bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, ensemble_size={}, bias={}'.format(
            self.in_features, self.out_features, self.ensemble_size, self.bias is not None
        )


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert np.prod(obs_shape) == 6 * 64 * 64, 'input must be vectorized image of size (6,64,64)'
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(6, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        if obs.ndim == 1:
            obs = obs.view(6,64,64)
            h = self.convnet(obs)
            h = h.view(self.repr_dim)
        elif obs.ndim == 2:
            obs = obs.view(-1,6,64,64)
            h = self.convnet(obs)
            h = h.view(-1, self.repr_dim)
        return h


class AtariEncoder(nn.Module):
    def __init__(self, obs_shape, drop_rate=0.0, layer_norm=False):
        super().__init__()

        assert np.prod(obs_shape) == 6 * 64 * 64, 'input must be vectorized image of size (6,64,64)'
        self.repr_dim = 64 * 4 * 4

        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=8, stride=4),
            nn.Identity() if drop_rate == 0.0 else nn.Dropout(p=drop_rate),
            nn.Identity() if not layer_norm else nn.LayerNorm((32,15,15)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.Identity() if drop_rate == 0.0 else nn.Dropout(p=drop_rate),
            nn.Identity() if not layer_norm else nn.LayerNorm((64,6,6)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.Identity() if drop_rate == 0.0 else nn.Dropout(p=drop_rate),
            nn.Identity() if not layer_norm else nn.LayerNorm((64,4,4)),
            nn.ReLU(inplace=True),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        if obs.ndim == 1:
            obs = obs.view(6,64,64)
            h = self.convnet(obs)
            h = h.view(self.repr_dim)
        elif obs.ndim == 2:
            obs = obs.view(-1,6,64,64)
            h = self.convnet(obs)
            h = h.view(-1, self.repr_dim)
        return h
    

class AtariSharedEncoder(nn.Module):
    def __init__(self, obs_shape, drop_rate=0.0, layer_norm=False):
        super().__init__()

        assert np.prod(obs_shape) == 6 * 64 * 64, 'input must be vectorized image of size (6,64,64)'
        self.repr_dim = 64 * 4 * 4 * 2

        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),
            nn.Identity() if drop_rate == 0.0 else nn.Dropout(p=drop_rate),
            nn.Identity() if not layer_norm else nn.LayerNorm((32,15,15)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.Identity() if drop_rate == 0.0 else nn.Dropout(p=drop_rate),
            nn.Identity() if not layer_norm else nn.LayerNorm((64,6,6)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.Identity() if drop_rate == 0.0 else nn.Dropout(p=drop_rate),
            nn.Identity() if not layer_norm else nn.LayerNorm((64,4,4)),
            nn.ReLU(inplace=True),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        if obs.ndim == 1:
            obs = obs.view(6,64,64)
            h_s = self.convnet(obs[:3,:,:])
            h_g = self.convnet(obs[3:,:,:])
            h_s = h_s.view(self.repr_dim//2)
            h_g = h_g.view(self.repr_dim//2)
        elif obs.ndim == 2:
            obs = obs.view(-1,6,64,64)
            h_s = self.convnet(obs[:,:3,:,:])
            h_g = self.convnet(obs[:,3:,:,:])
            h_s = h_s.view(-1, self.repr_dim//2)
            h_g = h_g.view(-1, self.repr_dim//2)
        return torch.concat([h_s, h_g], axis=-1)


class DDPGActor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        action_dim = action_shape[0]

        self.policy = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim),
                                    nn.Tanh(),
                                    nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_dim))
        self.apply(utils.weight_init)

    def forward(self, obs, std=None):
        assert std != None

        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)

        return dist


class SACActor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, log_std_bounds):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        action_dim = action_shape[0] * 2

        self.policy = nn.Sequential(# convert image/state to a normalized vector 
                                    nn.Linear(repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim),
                                    nn.Tanh(),
                                    # policy layers
                                    nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_dim))
        self.apply(utils.weight_init)

    def forward(self, obs):
        mu, log_std = self.policy(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        # TODO: switched to simple clipping instead of going the tanh / rescaling route
        log_std_min, log_std_max = self.log_std_bounds
        # log_std = torch.tanh(log_std)
        # log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
        log_std = torch.clip(log_std, log_std_min, log_std_max)
        std_pred = log_std.exp()

        dist = utils.SquashedNormal(mu, std_pred)

        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim),
                                   nn.Tanh())
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)
        
    def forward(self, obs, action):
        h_action = torch.cat([self.trunk(obs), action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class CriticEnsemble(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, num_ensemble=1):
        super().__init__()
        self.num_ensemble = num_ensemble

        self.critics = nn.ModuleList([Critic(repr_dim, action_shape, feature_dim, hidden_dim) for _ in range(num_ensemble)])

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        q1_list = []
        q2_list = []
        for critic in self.critics:
            q1, q2 = critic(obs, action)
            q1_list.append(q1)
            q2_list.append(q2)
        q1 = torch.concat(q1_list, dim=-1)
        q2 = torch.concat(q2_list, dim=-1)

        return q1, q2


class CriticParallelizedEnsemble(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, num_ensemble=1, drop_rate=0.0, layer_norm=False):
        super().__init__()
        self.num_ensemble = num_ensemble

        self.trunks = nn.Sequential(
            ParallelizedLinear(repr_dim, feature_dim, num_ensemble),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
        )
        self.Qs = nn.Sequential(
            ParallelizedLinear(feature_dim + action_shape[0], hidden_dim, num_ensemble),
            nn.Identity() if drop_rate == 0.0 else nn.Dropout(p=drop_rate),
            nn.Identity() if not layer_norm else nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            ParallelizedLinear(hidden_dim, hidden_dim, num_ensemble),
            nn.Identity() if drop_rate == 0.0 else nn.Dropout(p=drop_rate),
            nn.Identity() if not layer_norm else nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            ParallelizedLinear(hidden_dim, 1, num_ensemble)
        )

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        # repeat h to make amenable to parallelization
        dim = len(obs.shape)
        while dim < 3:
            obs = obs.unsqueeze(0)
            dim = len(obs.shape)
        dim = len(action.shape)
        while dim < 3:
            action = action.unsqueeze(0)
            dim = len(action.shape)
        obs = obs.expand(self.num_ensemble,-1,-1)
        action = action.expand(self.num_ensemble,-1,-1)

        h_action = torch.cat([self.trunks(obs), action], dim=-1)
        q = self.Qs(h_action)
        q = q.squeeze(1) # if original dim was 1D, squeeze the extra created layer, otherwise leave it alone

        return q # output is (ensemble_size, batch_size, output_size)