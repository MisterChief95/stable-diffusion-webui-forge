import dataclasses
from math import atan, pi

import k_diffusion
import numpy as np
import torch
from scipy import stats

from modules import shared


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / sigma


k_diffusion.sampling.to_d = to_d


@dataclasses.dataclass
class Scheduler:
    name: str
    label: str
    function: any

    default_rho: float = -1
    need_inner_model: bool = False
    aliases: list = None


def uniform(n, sigma_min, sigma_max, inner_model, device):
    return inner_model.get_sigmas(n).to(device)


def sgm_uniform(n, sigma_min, sigma_max, inner_model, device):
    start = inner_model.sigma_to_t(torch.tensor(sigma_max))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min))
    sigs = [inner_model.t_to_sigma(ts) for ts in torch.linspace(start, end, n + 1)[:-1]]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def get_align_your_steps_sigmas(n, sigma_min, sigma_max, device):
    # https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html
    def loglinear_interp(t_steps, num_steps):
        """
        Performs log-linear interpolation of a given array of decreasing numbers.
        """
        xs = np.linspace(0, 1, len(t_steps))
        ys = np.log(t_steps[::-1])

        new_xs = np.linspace(0, 1, num_steps)
        new_ys = np.interp(new_xs, xs, ys)

        interped_ys = np.exp(new_ys)[::-1].copy()
        return interped_ys

    if shared.sd_model.is_sdxl:
        sigmas = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029]
    else:
        # Default to SD 1.5 sigmas.
        sigmas = [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029]

    if n != len(sigmas):
        sigmas = np.append(loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)

    return torch.FloatTensor(sigmas).to(device)


def kl_optimal(n, sigma_min, sigma_max, device):
    alpha_min = torch.arctan(torch.tensor(sigma_min, device=device))
    alpha_max = torch.arctan(torch.tensor(sigma_max, device=device))
    step_indices = torch.arange(n + 1, device=device)
    sigmas = torch.tan(step_indices / n * alpha_min + (1.0 - step_indices / n) * alpha_max)
    return sigmas


def simple_scheduler(n, sigma_min, sigma_max, inner_model, device):
    sigs = []
    ss = len(inner_model.sigmas) / n
    for x in range(n):
        sigs += [float(inner_model.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def normal_scheduler(n, sigma_min, sigma_max, inner_model, device, sgm=False, floor=False):
    start = inner_model.sigma_to_t(torch.tensor(sigma_max))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min))

    if sgm:
        timesteps = torch.linspace(start, end, n + 1)[:-1]
    else:
        timesteps = torch.linspace(start, end, n)

    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(inner_model.t_to_sigma(ts))
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def ddim_scheduler(n, sigma_min, sigma_max, inner_model, device):
    sigs = []
    ss = max(len(inner_model.sigmas) // n, 1)
    x = 1
    while x < len(inner_model.sigmas):
        sigs += [float(inner_model.sigmas[x])]
        x += ss
    sigs = sigs[::-1]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def beta_scheduler(n, sigma_min, sigma_max, inner_model, device):
    # From "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et. al, 2024) """
    alpha = shared.opts.beta_dist_alpha
    beta = shared.opts.beta_dist_beta
    timesteps = 1 - np.linspace(0, 1, n)
    timesteps = [stats.beta.ppf(x, alpha, beta) for x in timesteps]
    sigmas = [sigma_min + (x * (sigma_max - sigma_min)) for x in timesteps]
    sigmas += [0.0]
    return torch.FloatTensor(sigmas).to(device)


def turbo_scheduler(n, sigma_min, sigma_max, inner_model, device):
    unet = inner_model.inner_model.forge_objects.unet
    timesteps = torch.flip(torch.arange(1, n + 1) * float(1000.0 / n) - 1, (0,)).round().long().clip(0, 999)
    sigmas = unet.model.predictor.sigma(timesteps)
    sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
    return sigmas.to(device)


def get_bong_tangent_sigmas(steps, slope, pivot, start, end):
    smax = ((2 / pi) * atan(-slope * (0 - pivot)) + 1) / 2
    smin = ((2 / pi) * atan(-slope * ((steps - 1) - pivot)) + 1) / 2

    srange = smax - smin
    sscale = start - end

    sigmas = [((((2 / pi) * atan(-slope * (x - pivot)) + 1) / 2) - smin) * (1 / srange) * sscale + end for x in range(steps)]

    return sigmas


def bong_tangent_scheduler(n, sigma_min, sigma_max, device, *, start=1.0, middle=0.5, end=0.0, pivot_1=0.6, pivot_2=0.6, slope_1=0.2, slope_2=0.2, pad=False):
    """https://github.com/ClownsharkBatwing/RES4LYF/blob/main/sigmas.py#L4076"""
    n += 2

    midpoint = int((n * pivot_1 + n * pivot_2) / 2)
    pivot_1 = int(n * pivot_1)
    pivot_2 = int(n * pivot_2)

    slope_1 = slope_1 / (n / 40)
    slope_2 = slope_2 / (n / 40)

    stage_2_len = n - midpoint
    stage_1_len = n - stage_2_len

    tan_sigmas_1 = get_bong_tangent_sigmas(stage_1_len, slope_1, pivot_1, start, middle)
    tan_sigmas_2 = get_bong_tangent_sigmas(stage_2_len, slope_2, pivot_2 - stage_1_len, middle, end)

    tan_sigmas_1 = tan_sigmas_1[:-1]
    if pad:
        tan_sigmas_2 = tan_sigmas_2 + [0]

    tan_sigmas = torch.tensor(tan_sigmas_1 + tan_sigmas_2)

    return tan_sigmas.to(device)


schedulers = [
    Scheduler("automatic", "Automatic", None),
    Scheduler("uniform", "Uniform", uniform, need_inner_model=True),
    Scheduler("karras", "Karras", k_diffusion.sampling.get_sigmas_karras, default_rho=7.0),
    Scheduler("exponential", "Exponential", k_diffusion.sampling.get_sigmas_exponential),
    Scheduler("polyexponential", "Polyexponential", k_diffusion.sampling.get_sigmas_polyexponential, default_rho=1.0),
    Scheduler("sgm_uniform", "SGM Uniform", sgm_uniform, need_inner_model=True, aliases=["SGMUniform"]),
    Scheduler("kl_optimal", "KL Optimal", kl_optimal),
    Scheduler("align_your_steps", "Align Your Steps", get_align_your_steps_sigmas),
    Scheduler("simple", "Simple", simple_scheduler, need_inner_model=True),
    Scheduler("normal", "Normal", normal_scheduler, need_inner_model=True),
    Scheduler("ddim", "DDIM", ddim_scheduler, need_inner_model=True),
    Scheduler("beta", "Beta", beta_scheduler, need_inner_model=True),
    Scheduler("turbo", "Turbo", turbo_scheduler, need_inner_model=True),
    Scheduler("bong_tangent", "Bong Tangent", bong_tangent_scheduler),
]

schedulers_map = {**{x.name: x for x in schedulers}, **{x.label: x for x in schedulers}}
