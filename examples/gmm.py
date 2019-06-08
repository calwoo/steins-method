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
x = torch.tensor([2.0], requires_grad=True)

def p(x):
    gauss1 = torch.exp(torch.distributions.Normal(-2,1).log_prob(x))
    gauss2 = torch.exp(torch.distributions.Normal(2,1).log_prob(x))
    return 1.0/3 * gauss1 + 2.0/3 * gauss2

### get particles and descenter
particles = differentiable(torch.distributions.Normal(-10,1).sample((50,1)))
rbf_kernel = RBFKernel(sigma=0.05)
plot_dist(particles)

descent = SVGD(rbf_kernel, p)
descent.seed(particles)

# plot_dist(descent.particles)

### run descent
descent.train(lr=0.05, epochs=300)
plot_dist(descent.particles)
