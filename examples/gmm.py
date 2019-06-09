"""
Toy example on 1D gaussian mixture model, taken from

[Q. Liu, D. Wang] Stein variational gradient descent: 
a general purpose bayesian inference algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

from stein.kernels import *
from stein.operators import *
from stein.utils import *
from stein.descent import SVGD


### data
def p(x):
    gauss1 = torch.exp(torch.distributions.Normal(-2,1).log_prob(x))
    gauss2 = torch.exp(torch.distributions.Normal(2,1).log_prob(x))
    return 1.0/3 * gauss1 + 2.0/3 * gauss2

### get particles and descenter
particles = differentiable(torch.distributions.Normal(-10,1).sample((200,1)))
rbf_kernel = RBFKernel(sigma=1.0)
#plot_dist(p, particles)

descent = SVGD(rbf_kernel, p)
descent.seed(particles)

"""
other_pcls = differentiable(particles)
K = rbf_kernel.eval(particles, other_pcls)
logp_pcls = descent.KSD.logp(particles)
grad_logp = grad(logp_pcls, particles, grad_outputs=torch.ones_like(logp_pcls))
grad_K = grad(K, particles, grad_outputs=torch.ones_like(K))

"""
# plot_dist(descent.particles)

### run descent
descent.train(lr=0.5, epochs=500, gif=True)
plot_dist(p, descent.particles)
