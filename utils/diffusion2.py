import numpy as np
import torch
import torch.nn as nn
from utils.parsers import parser
import time
from utils.model import *

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          z_0,
                          z_t,
                          keepdim=False):

    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()

    output = model(x, t, z_0, z_t)
    #output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2))
    else:
        return (e - output).square().sum(dim=(1, 2)).mean(dim=0)


class Diffusion(nn.Module):
    def __init__(self, args, device= None):
        super(Diffusion, self).__init__()
        ###input
        self.batch_size = args.b_size
        self.max_seq = args.max_seq
        self.emb_dim = args.emb_dim
        self.z_dim = args.z_dim
        self.device = device
        #diffusion
        self.model_var_type = args.diff_var_type
        self.beta_schedule = args.noise_schedule
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end
        self.num_diffusion_timesteps =args.diff_steps



        self.model = AttnBlock(self.emb_dim, self.z_dim).to(self.device)   #模型怎么修改

        betas = get_beta_schedule(
            beta_schedule= self.beta_schedule,
            beta_start= self.beta_start,
            beta_end= self.beta_end,
            num_diffusion_timesteps= self.num_diffusion_timesteps
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train_loss(self, node_input, z_0 , z_t):

        #input_reshape = input.reshape(-1, self.emb_dim)
        node_input = node_input.permute(0, 2, 1)

        n = node_input.size(0)
        e = torch.randn_like(node_input)
        b = self.betas

        # antithetic sampling
        t = torch.randint(
            low=0, high=self.num_timesteps, size=(n // 2 + 1,)
        ).to(self.device)

        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

        loss = noise_estimation_loss(self.model, node_input, t, e, b, z_0, z_t)
        return loss


    def sample(self, input, z_0, z_t):

        input = input.permute(0, 2, 1)

        with torch.no_grad():
            x = torch.randn_like(input)
            x = self.sample_image(x, self.model, z_0, z_t)
        x = x.permute(0, 2, 1)
        return x

    def sample_image(self, x, model, z_0, z_t, last=True):

        skip = self.num_timesteps // self.num_timesteps
        seq = range(0, self.num_timesteps, skip)

        from utils.denoising import generalized_steps

        xs = generalized_steps(x, seq, model, self.betas, z_0, z_t, eta=0.0)
        x = xs

        # if last:
        #     x = x[0][-1]
        return x

    def test(self):
        pass

if __name__ == "__main__":

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 10
    input = torch.randn(batch_size, args.max_seq, args.z_dim).to("cuda:0")
    z = torch.randn(batch_size, args.z_dim).to("cuda:0")
    # out = torch.cat([input, z],)

    first_point_enc = z.unsqueeze(1)
    diff_input = torch.cat([first_point_enc, input], 1)

    a = torch.randn_like(diff_input)
    mask = a.ge(0.5)

    not_mask = ~mask
    diff_model_input = diff_input * mask

    diff_model_input = diff_model_input.permute(0, 2, 1)

    time_step = torch.linspace(0, 10, steps=100, out=None).to("cuda:0")

    model = Diffusion(args, device)
    model = model.to("cuda:0")

    start_time = time.time()
    for i in range(0,10):
        x = model.train_loss(diff_model_input, z)
        print("x", x)
    end_time = time.time()
    print('Finished! Time used: {:.3f}mins.'.format((end_time-start_time)/60))

    print(0)
