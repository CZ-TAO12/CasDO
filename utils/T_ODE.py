import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import relu

from torch.distributions.normal import Normal
from utils.parsers import parser
from utils.losses import *
from utils.diffeq_solver import DiffeqSolver
from utils.lib import *
from utils.ode_func import ODEFunc


def sample_standard_gaussian(mu, sigma):
	device = get_device(mu)

	d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
	r = d.sample(mu.size()).squeeze(-1)
	return r * sigma.float() + mu.float()


class GRU_unit(nn.Module):
    def __init__(self, latent_dim, input_dim,
                 update_gate=None,
                 reset_gate=None,
                 new_state_net=None,
                 n_units=100,
                 device=torch.device("cpu")):
        super(GRU_unit, self).__init__()

        if update_gate is None:
            self.update_gate = nn.Sequential(
                nn.Linear(latent_dim * 2 + input_dim, n_units),
                nn.Tanh(),
                nn.Linear(n_units, latent_dim),
                nn.Sigmoid())
            init_network_weights(self.update_gate)
        else:
            self.update_gate = update_gate

        if reset_gate is None:
            self.reset_gate = nn.Sequential(
                nn.Linear(latent_dim * 2 + input_dim, n_units),
                nn.Tanh(),
                nn.Linear(n_units, latent_dim),
                nn.Sigmoid())
            init_network_weights(self.reset_gate)
        else:
            self.reset_gate = reset_gate

        if new_state_net is None:
            self.new_state_net = nn.Sequential(
                nn.Linear(latent_dim * 2 + input_dim, n_units),
                nn.Tanh(),
                nn.Linear(n_units, latent_dim * 2))
            init_network_weights(self.new_state_net)
        else:
            self.new_state_net = new_state_net


    def forward(self, y_mean, y_std, x, masked_update=True):
        y_concat = torch.cat([y_mean, y_std, x], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)

        new_state, new_state_std = split_last_dim(self.new_state_net(concat))
        new_state_std = new_state_std.abs()

        new_y = (1 - update_gate) * new_state + update_gate * y_mean
        new_y_std = (1 - update_gate) * new_state_std + update_gate * y_std

        assert (not torch.isnan(new_y).any())
        assert (not torch.isnan(new_y_std).any())

        new_y_std = new_y_std.abs()
        return new_y, new_y_std


class t_ode(nn.Module):
    # Derive z0 by running ode backwards.
    # For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
    # Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
    # Continue until we get to z0
    def __init__(self, input_dim, latent_dim, z0_diffeq_solver=None,
                 z0_dim=None, GRU_update=None,
                 n_gru_units=128,
                 device=torch.device("cpu")):

        super(t_ode, self).__init__()

        if z0_dim is None:
            self.z0_dim = latent_dim
        else:
            self.z0_dim = z0_dim

        if GRU_update is None:
            self.GRU_update = GRU_unit(latent_dim, input_dim,
                                       n_units=n_gru_units,
                                       device=device).to(device)

            self.z_gru = GRU_unit(latent_dim, input_dim,
                                       n_units=n_gru_units,
                                       device=device).to(device)

        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device

        self.transform_z0 = nn.Sequential(
            nn.Linear(latent_dim * 2, 100),
            nn.Tanh(),
            nn.Linear(100, self.z0_dim * 2)
        )
        init_network_weights(self.transform_z0)

    def forward(self, data, time_steps, run_backwards=True):
        # data, time_steps -- observations and their time stamps
        # IMPORTANT: assumes that 'data' already has mask concatenated to it
        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        n_traj, n_tp, n_dims = data.size()

        last_yi, last_yi_std, latent_ys = self.run_odernn(
            data, time_steps, run_backwards=run_backwards)

        means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
        std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)

        mean_z0, std_z0 = split_last_dim(self.transform_z0(torch.cat((means_z0, std_z0), -1)))
        std_z0 = std_z0.abs()
        return mean_z0, std_z0, last_yi

    def run_odernn(self, data, time_steps, run_backwards=True):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it

        n_traj, n_tp, n_dims = data.size()
        n_batch, n_step = time_steps.size()

        device = get_device(data)

        prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)


        if run_backwards:
            prev_t, t_i = time_steps[:,-1] + 0.01, time_steps[:,-1]
        else:
            prev_t, t_i = time_steps[:,0], time_steps[:,0] + 0.01

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())



        latent_ys = []
        # Run ODE backwards and combine the y(t) estimates using gating
        time_points_iter = range(0, n_step)
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        for i in time_points_iter:

            t_gap = torch.abs(t_i - prev_t).reshape(-1, 1)

            inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y) * t_gap
            assert (not torch.isnan(inc).any())

            ode_sol = prev_y + inc
            ode_sol = torch.stack((prev_y, ode_sol), 2).to(device)  #shape [1,10,2,64]

            assert (not torch.isnan(ode_sol).any())

            yi_ode = ode_sol[:, :, -1, :]
            xi = data[:, i, :].unsqueeze(0)

            yi, yi_std = self.GRU_update(yi_ode, prev_std, xi)
            zi, zi_std = self.z_gru(prev_y, prev_std, xi)

            u = 1 / torch.exp(t_gap)

            yi = u * yi + (1-u) * zi
            yi_std = u * yi_std + (1-u) * zi_std
            prev_y, prev_std = yi, yi_std

            if run_backwards:
                prev_t, t_i = time_steps[:, i] , time_steps[:, i-1]
            else:
                if i == 0:
                    prev_t, t_i = time_steps[:, 0] + 0.01, time_steps[:, 1]
                elif i < (n_step-2):
                    prev_t, t_i = time_steps[:, i] + 0.01, time_steps[:, i+1]

            latent_ys.append(yi)

        latent_ys = torch.stack(latent_ys, 1)

        assert (not torch.isnan(yi).any())
        assert (not torch.isnan(yi_std).any())

        return yi, yi_std, latent_ys

if __name__ == "__main__":
    print(0)
