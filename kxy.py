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
X = differentiable(torch.distributions.Normal(0,1).sample((500, 10)))

x = torch.tensor([[1.0, 2.0], [0.0, -1.0]], requires_grad=True)
y = differentiable(X)

kxy = rbf.eval(X, y)
print(grad(kxy, X, grad_outputs=torch.ones_like(kxy))[0])

### direct vectorized grad_x k(x,y)
sigma = rbf.sigma
_dxkxy = -torch.mm(kxy, X)
sumkxy = torch.sum(kxy, 1, keepdim=True)
dxkxy = _dxkxy + X * sumkxy
print(dxkxy)

