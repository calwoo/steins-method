import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from kernels import *
from stein import *


def p(x):
    """
    torch tensor version of N(0,1) gaussian
    """
    return torch.exp(torch.distributions.Normal(0,1).log_prob(x))

q = torch.distributions.Normal(0,1)

ksd = KSD(rbf_kernel, p)
ksd.eval(q, num_samples=10)