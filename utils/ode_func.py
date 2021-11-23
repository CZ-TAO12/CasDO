import torch
import torch.nn as nn
import utils.lib as utils

class ODEFunc(nn.Module):
    def __init__(self, input_dim, device = torch.device("cpu"), units = 128):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc, self).__init__()

        self.input_dim = input_dim
        self.ode_units = units

        self.ode_func_net = torch.nn.Sequential(
            nn.Linear(self.input_dim, self.ode_units),
            nn.Tanh(),
            nn.Linear(self.ode_units, self.ode_units),
            nn.Tanh(),
            nn.Linear(self.ode_units, self.input_dim)
        )
        utils.init_network_weights(self.ode_func_net)

        self.ode_func_net = self.ode_func_net.to(device)
        self.device = device
        self.gradient_net = self.ode_func_net

    def forward(self, t_local, y, backwards = False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        return self.gradient_net(y)

    def sample_next_point_from_prior(self, t_local, y):
        """
        t_local: current time point
        y: value at the current time point
        """
        return self.get_ode_gradient_nn(t_local, y)





