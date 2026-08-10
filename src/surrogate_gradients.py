#
# surrogate_gradients.py
#
# The original sparch package (Bittar & Garner, 2022) only ships the Boxcar
# surrogate (`SpikeFunctionBoxcar`). Experiment 2 of the thesis compares three
# additional surrogate gradient functions -- Exponential, Gaussian, and
# Multi-Gaussian. This module defines all
# four as `torch.autograd.Function` subclasses with an identical interface,
# so any of them can be swapped into the existing *Layer classes with a
# single string argument.
#
# All four share the same forward pass (a hard Heaviside spike), and differ
# only in what gradient they substitute for the (formally zero-almost-
# everywhere) true derivative during backward().
#
"""Surrogate gradient functions for training spiking neurons via backprop."""

import math

import torch


class SpikeFunctionBoxcar(torch.autograd.Function):
    """
    Boxcar surrogate gradient (DECOLLE-style, Kaiser et al. 2020).
    Constant gradient of 1 within [-0.5, 0.5], zero elsewhere.
    This is the default used in the original sparch codebase / Experiment 1.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        grad_x = grad_spikes.clone()
        grad_x[x <= -0.5] = 0
        grad_x[x > 0.5] = 0
        return grad_x


class SpikeFunctionExponential(torch.autograd.Function):
    """
    Exponential surrogate gradient (SLAYER-style, Shrestha & Orchard 2018).

        d(spike)/dx  ~=  alpha * exp(-beta * |x|)

    `alpha` scales the peak gradient magnitude and `beta` controls how
    sharply it decays away from the threshold. Both are fixed (non-trainable)
    hyperparameters of the surrogate itself, not of the neuron.
    """

    alpha = 1.0
    beta = 5.0

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        surrogate = SpikeFunctionExponential.alpha * torch.exp(
            -SpikeFunctionExponential.beta * torch.abs(x)
        )
        return grad_spikes * surrogate


class SpikeFunctionGaussian(torch.autograd.Function):
    """
    Gaussian surrogate gradient (Yin et al. 2021).

    `sigma` controls the width of the surrogate around the threshold; a
    smaller sigma gives a sharper (more boxcar-like) approximation, a larger
    sigma gives a smoother, longer-range gradient.
    """

    sigma = 0.5

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        sigma = SpikeFunctionGaussian.sigma
        surrogate = torch.exp(-(x ** 2) 
        return grad_spikes * surrogate


class SpikeFunctionMultiGaussian(torch.autograd.Function):
    """
    Multi-Gaussian surrogate gradient (Yin, Corradi & Bohte, 2021 -- the same
    paper that introduces the adaptive / recurrent-adaptive LIF neurons used
    elsewhere in this codebase). Instead of a single Gaussian lobe, this adds
    two small negative side-lobes on either side of a central positive lobe,
    which tends to give a cleaner gradient signal for adaptive neurons:

        g(x) = (1 + h) * N(x; 0, s) - h * N(x; s, 6s) - h * N(x; -s, 6s)

    where N(x; mu, sigma) is a Gaussian density centered at `mu` with std
    `sigma`. `h` sets the depth of the side lobes and `s` sets the base
    width, both fixed surrogate hyperparameters.
    """

    h = 0.15
    s = 0.5

    @staticmethod
    def _gaussian(x, mu, sigma):
        return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (
            sigma * math.sqrt(2 * math.pi)
        )

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        h = SpikeFunctionMultiGaussian.h
        s = SpikeFunctionMultiGaussian.s
        g = SpikeFunctionMultiGaussian._gaussian
        surrogate = (
            (1 + h) * g(x, 0.0, s)
            - h * g(x, s, 6 * s)
            - h * g(x, -s, 6 * s)
        )
        return grad_spikes * surrogate


# Registry so layers/experiments can select a surrogate by name, matching
# the four options discussed in the thesis ("box-car" default + the three
# compared in Experiment 2).
SURROGATE_FUNCTIONS = {
    "boxcar": SpikeFunctionBoxcar,
    "exponential": SpikeFunctionExponential,
    "gaussian": SpikeFunctionGaussian,
    "multigaussian": SpikeFunctionMultiGaussian,
}


def get_surrogate_fn(name: str):
    """Look up a surrogate gradient autograd.Function class by name."""
    key = name.lower()
    if key not in SURROGATE_FUNCTIONS:
        raise ValueError(
            f"Unknown surrogate '{name}'. Choose from {list(SURROGATE_FUNCTIONS)}"
        )
    return SURROGATE_FUNCTIONS[key]
