import torch
import  torch.nn as nn
from utils.T_ODE import t_ode
from utils.ode_func import ODEFunc
from utils.diffeq_solver import DiffeqSolver
from utils.lib import *
from utils.losses import *
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent
from utils.diffusion2 import *
from torch.distributions import kl_divergence, Independent

######prediction
class MLP(nn.Module):
    def __init__(self, inputdim, device):
        super(MLP, self).__init__()
        self.input = inputdim
        dense = torch.nn.Sequential(
            nn.Linear(self.input, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )
        self.dense = dense.to(device)
        init_network_weights(self.dense)

    def forward(self, input):
        output = self.dense(input)
        return output

class Decoder(nn.Module):
    def __init__(self, args, device):
        super(Decoder, self).__init__()

        self.input_dim = args.emb_dim
        self.z_dim = args.z_dim
        self.rnn_units = args.rnn_units
        self.device = device

        gru = nn.GRU(input_size= self.z_dim, hidden_size=self.rnn_units, batch_first=True, bidirectional=True)
        mlp = nn.Linear(self.rnn_units*2, self.input_dim)

        self.gru = gru.to(self.device)
        self.mlp = mlp.to(self.device)

        init_network_weights(self.mlp)

    def forward(self, data):
        out, hidden = self.gru(data)
        output = self.mlp(out)
        return output

class ODE_diffusion_LODE(nn.Module):
    def __init__(self,
                 args,
                 device= None):
        super(ODE_diffusion_LODE, self).__init__()

        self.args = args
        self.input_dim = args.emb_dim
        self.z_dim = args.z_dim
        self.ode_units = args.ode_units
        self.rnn_units = args.rnn_units
        self.max_seq = args.max_seq
        self.device = device
        self.MSE = torch.nn.MSELoss()

        self.z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

        rec_ode_func = ODEFunc(
            input_dim=self.z_dim,
            device=device,
            units = self.ode_units).to(self.device)

        self.z0_diffeq_solver = DiffeqSolver(rec_ode_func, "euler", odeint_rtol=1e-3, odeint_atol=1e-4,
                                             device=self.device).to(self.device)


        self.encoder_z0 = t_ode(self.input_dim, self.z_dim, self.z0_diffeq_solver,
                                             n_gru_units=self.rnn_units, device=self.device).to(self.device)

        self.gen_ode_func = ODEFunc(
            input_dim=self.z_dim,
            device=self.device).to(device)

        self.diffeq_solver = DiffeqSolver(self.gen_ode_func, 'euler', odeint_rtol=1e-3, odeint_atol=1e-4,
                                          device=self.device).to(self.device)


        self.gen_decode = Decoder(args,self.device).to(self.device)

        self.cas_gru = nn.GRU(input_size=self.z_dim, hidden_size=self.z_dim,
                               batch_first=True, bidirectional=True).to(self.device)

        self.dense = MLP(self.z_dim * 3, self.device).to(self.device)

    def encoder(self, input, time_steps, backwards=True):

        batch_size = input.size(0)

        first_point_mu, first_point_std, z_0 = self.encoder_z0(
            input, time_steps, run_backwards=backwards)

        z_0 = z_0.squeeze(dim=0)

        yi, yi_std, latent_ys = self.encoder_z0.run_odernn(input,
                                                           time_steps,
                                                           run_backwards=False
                                                           )
        yi = yi.squeeze(dim=0)
        latent_ys = latent_ys.squeeze(dim=0)
        latent_ys = latent_ys.permute(1, 0, 2)
        latent_ys = latent_ys.reshape(-1, self.max_seq, self.z_dim)
        out, hidden = self.cas_gru(latent_ys)
        z1, z2 = hidden[0], hidden[1]
        out = torch.cat((z1, z2), 1)

        return z_0, yi, out

    def decoder(self, diff_input , output, time_steps, backwards=True):
        batch_size = diff_input.size(0)

        first_point_mu, first_point_std, z_0 = self.encoder_z0(
            diff_input, time_steps, run_backwards=backwards)

        first_point_mu = first_point_mu.squeeze(dim=0)
        first_point_std = first_point_std.squeeze(dim=0)

        first_point_enc = sample_standard_gaussian(first_point_mu, first_point_std)
        first_point_std = first_point_std.abs()
        assert (torch.sum(first_point_std < 0) == 0.)
        assert (not torch.isnan(time_steps).any())
        assert (not torch.isnan(first_point_enc).any())

        ####generation
        gen_timestep = torch.linspace(0, 1, steps=100).to(self.device)
        sol_y = self.diffeq_solver(first_point_enc, gen_timestep)

        time_zeros = torch.zeros(size=(batch_size, 1)).to(self.device)
        time_steps2 = torch.cat([time_zeros, time_steps], 1).to(self.device)
        time_steps3 = time_steps2[:, 0:100]
        time_spacing = time_steps - time_steps3
        time_spacing[:, 0] = 1
        time_spacing = (time_spacing * 100).unsqueeze(2)

        em_zeros = torch.zeros(size=(batch_size, 1, self.z_dim)).to(self.device)
        sol_y2 = torch.cat([em_zeros, sol_y], 1).to(self.device)
        sol_y3 = sol_y2[:, 0:100, :]
        z_em = sol_y - sol_y3
        sol_y = z_em * time_spacing
        sol_y2 = torch.cat([em_zeros, sol_y], 1).to(self.device)
        sol_y3 = sol_y2[:, 0:100, :]
        pre_z = sol_y + sol_y3
        z_t = pre_z[:, 99, :]

        diff_pred_x = self.gen_decode(pre_z)

        # kl_loss
        fp_distr = Normal(first_point_mu, first_point_std)
        kl_loss = kl_divergence(fp_distr, self.z0_prior)
        kl_loss = torch.sum(kl_loss)
        diff_kl_loss = kl_loss / float(batch_size)

        com_out = torch.cat((output, z_t), 1)

        outputs = self.dense(com_out)

        gaussian = Independent(Normal(loc=outputs, scale=0.01), 1)

        return outputs, gaussian, diff_pred_x, diff_kl_loss

##########diffusion model
def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, eps=1e-6, affine=True)

class AttnBlock(nn.Module):
    def __init__(self, in_channels, z_dim):
        super(AttnBlock, self).__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.ch = 128
        self.temb_ch = self.ch * 4  #512

        self.temb_dense = torch.nn.Sequential(
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch)
        )

        self.temb_proj = torch.nn.Linear(
            self.temb_ch,
            self.in_channels
        )

        self.temb_proj2 = torch.nn.Linear(
            self.z_dim,
            self.in_channels
        )

        self.dropout = torch.nn.Dropout(0.1)
        self.norm1 = Normalize(in_channels)
        self.norm2 = Normalize(in_channels)
        self.norm = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     in_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.conv2 = torch.nn.Conv1d(in_channels,
                                     in_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x, t,  z_0, z_t):
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb_dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb_dense[1](temb)

        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        z_00 = self.temb_proj2(nonlinearity(z_0))[:, :, None]
        z_tt = self.temb_proj2(nonlinearity(z_t))[:, :, None]

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None] + z_00 + z_tt

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        x_1 = x + h

        h_ = x_1
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h = q.shape
        q = q.reshape(b, c, h)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h)

        h_ = self.proj_out(h_)

        return x_1 + h_

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps
        x_reshaped = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshaped)

        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.contiguous().view(-1, x.size(1), y.size(-1))
        return y



