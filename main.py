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


def p(x):
    """
    torch tensor version of N(0,1) gaussian
    """
    return torch.exp(torch.distributions.Normal(0,1).log_prob(x))

num_samples=10

q = torch.distributions.Normal(0,1)
samples = q.sample((num_samples,))

rbf_kernel = RBFKernel(sigma=1.0)
ksd = KSD(rbf_kernel, p)
print(ksd.eval(q, num_samples=10))