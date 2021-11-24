import argparse

"""
Dataset | Observation Time           | Prediction Time               |
---------------------------------------------------------------------|
weibo   | 3600 (1 hour)              | 3600*24 (86400, 1 day)        |
twitter | 3600*24*2 (172800, 2 days) | 3600*24*32 (2764800, 32 days) |
acm     | 3 (years)                  | 10 (years)                    |
aps     | 365*3 (1095, 3 years)      | 365*20+5 (7305, 20 years)     |
dblp    | 5 (years)                  | 20 (years)                    |
"""

parser = argparse.ArgumentParser()
######################  dataset path
parser.add_argument('--input', default='./dataset/twitter/', type=str, help="Dataset path.")
parser.add_argument('--gg_path', default='global_graph.pkl', type=str, help="Global graph path.")
######################gene_cas
parser.add_argument("--observation_time", type=int, default=3600*24*1, help="Observation time.")
parser.add_argument("--prediction_time", type=int, default=3600*24*32, help="Prediction time.")

######################gene_emb
parser.add_argument("--cg_emb_dim", type=int, default=40, help="Cascade graph embedding dimension.")
parser.add_argument("--gg_emb_dim", type=int, default=40, help="Global graph embedding dimension.")
parser.add_argument("--max_seq", type=int, default=100, help="Max length of cascade sequence.")
parser.add_argument("--num_s", type=int, default=2, help="Number of s for spectral graph wavelets.")

######################model
parser.add_argument("--lr", type= float, default=5e-4, help="Learning rate.")
parser.add_argument("--b_size", type=int, default=64, help="Batch size.")
parser.add_argument("--emb_dim", type=int, default=40+40, help="Embedding dimension (cascade emb_dim + global emb_dim")
parser.add_argument("--z_dim", type=int, default=64, help="Dimension of latent variable z.")
parser.add_argument("--rnn_units", type=int, default=128, help="Dimension of latent variable z.")
parser.add_argument("--n_flows", type=int, default=8, help="Number of NF transformations.")
parser.add_argument("--verbose", type=int, default=2, help="Verbose.")
parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
parser.add_argument("--epochs", type=int, default=1000, help="train epochs.")

#  ODE_diffusion
######################diffusion
parser.add_argument('--diff_mean_type', default='eps', type=str, help="diffusion model's mean type")           #xprev   xstart  eps
parser.add_argument('--diff_var_type', default='fixedlarge', type=str, help="diffusion model's variance type")  #'learned'
parser.add_argument('--diff_loss_type', default='mse', type=str, help="diffusion model's loss type")  #kl
parser.add_argument('--diff_steps', default=8, type=int, help="diffusion model's time step")
parser.add_argument('--noise_schedule', default="linear", type=str, help="diffusion model's time step")
parser.add_argument('--beta_start', default=0.0001, type=int, help="diffusion model's time step")
parser.add_argument('--beta_end', default=0.02, type=int, help="diffusion model's time step")
parser.add_argument('--ema', default=1, type=int, help="diffusion model's time step")
parser.add_argument('--log_path', default='./checkpoint', type=str, help="diffusion model's time step")
parser.add_argument('--diff_lr', default=0.0002, type=float, help="diffusion model's time step")

######################ODE
parser.add_argument('--ode_units', type=int, default=100, help="Number of units per layer in ODE func")
