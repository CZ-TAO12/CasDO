"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def get_device(tensor):
	device = torch.device("cpu")
	if tensor.is_cuda:
		device = tensor.get_device()
	return device

def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std, indices = None):
	n_data_points = mu_2d.size()[-1]

	if n_data_points > 0:
		gaussian = Independent(Normal(loc = mu_2d, scale = obsrv_std.repeat(n_data_points)), 1)
		log_prob = gaussian.log_prob(data_2d)
		log_prob = log_prob / n_data_points
	else:
		log_prob = torch.zeros([1]).to(get_device(data_2d)).squeeze()
	return log_prob


def compute_masked_likelihood(mu, data, mask, likelihood_func):
    # Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
    n_traj_samples, n_traj, n_timepoints, n_dims = data.size()

    res = []
    for i in range(n_traj_samples):
        for k in range(n_traj):
            for j in range(n_dims):
                data_masked = torch.masked_select(data[i, k, :, j], mask[i, k, :, j].bool())

                # assert(torch.sum(data_masked == 0.) < 10)

                mu_masked = torch.masked_select(mu[i, k, :, j], mask[i, k, :, j].bool())
                log_prob = likelihood_func(mu_masked, data_masked, indices=(i, k, j))
                res.append(log_prob)
    # shape: [n_traj*n_traj_samples, 1]

    res = torch.stack(res, 0).to(get_device(data))
    res = res.reshape((n_traj_samples, n_traj, n_dims))
    # Take mean over the number of dimensions
    res = torch.mean(res, -1)  # !!!!!!!!!!! changed from sum to mean
    res = res.transpose(0, 1)
    return res

def masked_gaussian_log_density(mu, data, obsrv_std, mask=None):
    # these cases are for plotting through plot_estim_density
    if (len(mu.size()) == 3):
        # add additional dimension for gp samples
        mu = mu.unsqueeze(0)

    if (len(data.size()) == 2):
        # add additional dimension for gp samples and time step
        data = data.unsqueeze(0).unsqueeze(2)
    elif (len(data.size()) == 3):
        # add additional dimension for gp samples
        data = data.unsqueeze(0)

    n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

    assert (data.size()[-1] == n_dims)

    # Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
    if mask is None:
        mu_flat = mu.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)
        n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
        data_flat = data.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)

        res = gaussian_log_likelihood(mu_flat, data_flat, obsrv_std)
        res = res.reshape(n_traj_samples, n_traj).transpose(0, 1)
    else:
        # Compute the likelihood per patient so that we don't priorize patients with more measurements
        func = lambda mu, data, indices: gaussian_log_likelihood(mu, data, obsrv_std=obsrv_std, indices=indices)
        res = compute_masked_likelihood(mu, data, mask, func)
    return res

def get_gaussian_likelihood(truth, pred_y, mask=None):
    # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
    # truth shape  [n_traj, n_tp, n_dim]
    n_traj, n_tp, n_dim = truth.size()

    # Compute likelihood of the data under the predictions
    truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)

    if mask is not None:
        mask = mask.repeat(pred_y.size(0), 1, 1, 1)
    log_density_data = masked_gaussian_log_density(pred_y, truth_repeated,
                                                   obsrv_std=0.01, mask=mask)
    log_density_data = log_density_data.permute(1, 0)
    log_density = torch.mean(log_density_data, 1)

    # shape: [n_traj_samples]
    return log_density

def mse(mu, data, indices = None):
	n_data_points = mu.size()[-1]

	if n_data_points > 0:
		mse = nn.MSELoss()(mu, data)
	else:
		mse = torch.zeros([1]).to(get_device(data)).squeeze()
	return mse

def compute_mse(mu, data, mask = None):
	# these cases are for plotting through plot_estim_density
	if (len(mu.size()) == 3):
		# add additional dimension for gp samples
		mu = mu.unsqueeze(0)

	if (len(data.size()) == 2):
		# add additional dimension for gp samples and time step
		data = data.unsqueeze(0).unsqueeze(2)
	elif (len(data.size()) == 3):
		# add additional dimension for gp samples
		data = data.unsqueeze(0)

	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
	assert(data.size()[-1] == n_dims)

	# Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
	if mask is None:
		mu_flat = mu.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
		n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
		data_flat = data.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
		res = mse(mu_flat, data_flat)
	else:
		# Compute the likelihood per patient so that we don't priorize patients with more measurements
		res = compute_masked_likelihood(mu, data, mask, mse)
	return res

def get_mse(truth, pred_y, mask=None):
    # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
    # truth shape  [n_traj, n_tp, n_dim]
    n_traj, n_tp, n_dim = truth.size()

    # Compute likelihood of the data under the predictions
    truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)

    if mask is not None:
        mask = mask.repeat(pred_y.size(0), 1, 1, 1)

    # Compute likelihood of the data under the predictions
    log_density_data = compute_mse(pred_y, truth_repeated, mask=mask)
    # shape: [1]
    return torch.mean(log_density_data)


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}
