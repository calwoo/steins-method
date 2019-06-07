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

### langevin-stein operator
def langevin_stein_g(p, phi, x):
    """
    implementation of the Langevin-Stein operator, which is an operator
    variational objective useful for SVGD.
    """
    def logp(x):
        return torch.log(p(x))
    nabla_logp = grad(logp(x), x, create_graph=True)[0]
    return torch.dot(phi(x), nabla_logp) + grad(phi(x), x, create_graph=True)[0]
   
"""
num_samples=10

q = torch.distributions.Normal(0,1)
samples = q.sample((num_samples,))

rbf_kernel = RBFKernel(sigma=1.0)
ksd = KSD(rbf_kernel, p)
# ksd.eval(q, num_samples=10)

phi = ksd.optimal_fn(q, samples)

def kern_curry(y):
    def k(x):
        return ksd.kernel.eval(x,y).view(-1)
    return k

### test
x = torch.tensor([2.0], requires_grad=True)
y = langevin_stein(ksd.p, kern_curry(torch.tensor([1.0])), x.view(-1))
z = langevin_stein_g(ksd.p, kern_curry(torch.tensor([1.0])), x.view(-1))

y.backward()
z.backward()
print(y, x.grad)
print(z, x.grad)
"""

x = torch.tensor([2.0], requires_grad=True)

def f(x):
    log_prob = torch.log(p(x))
    log_prob.backward()
    print(x.grad)
    return log_prob

#y = f(x)
#print(y)
#y.backward()
#print(x.grad)

x_clone = x.clone()

print(x, x_clone)

y = x**2
y.backward()

print(x.grad, x_clone.grad)