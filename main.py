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


### get particles and descenter
particles = torch.linspace(-2,2,steps=20, requires_grad=True).view(-1,1)
rbf_kernel = RBFKernel(sigma=1.0)

def p(x):
    log_prob = torch.distributions.Normal(0,1).log_prob(x)
    return torch.exp(log_prob)

descent = SVGD(rbf_kernel, p)
descent.seed(particles)

# plot_dist(descent.particles)

### run descent
descent.train(lr=1e-2, epochs=100)
plot_dist(descent.particles)
