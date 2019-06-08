import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

from stein.kernels import *
from stein.operators import *
from stein.descent import *
from stein.utils import *


rbf = RBFKernel(sigma=1.0)
X = torch.distributions.Normal(0,1).sample((500, 10))
particles = torch.tensor([[1.0, 2.0], [0.0, -1.0]], requires_grad=True)

def dists(X, Y):
    """
    completely vectorized squared-dist function in numpy
    """
    dists = torch.sum(X**2, 1).unsqueeze(-1) + torch.sum(Y**2, 1) -\
        2*torch.mm(X, Y.t())
    return dists

@timed
def ker(X, Y):
    dist = dists(X, Y)
    return torch.exp(-dist / 2)

def p(x):
    return torch.exp(torch.distributions.Normal(0,1).log_prob(x))

def logp(x):
    return torch.log(p(x))

### langevin-stein
ksd = KSD(rbf, p)
phi = ksd.optimal_fn(particles)
# print(langevin_stein(p, phi, particles[0]))

def ls(logp, kernel, particles):
    K = kernel.eval(particles, particles)
    grad_logp = grad(logp(particles), particles, grad_outputs=torch.ones_like(particles))[0]
    term1 = torch.mm(K, grad_logp)
    return term1

print(ls(logp, rbf, particles))

### vectorized langevin-stein

y = logp(particles)
grad_logp = grad(y, particles, grad_outputs=torch.ones_like(y))



