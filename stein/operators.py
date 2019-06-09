### imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.style.use("ggplot")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

from stein.utils import *

### langevin-stein operator
def langevin_stein(p, phi, x):
    """
    implementation of the Langevin-Stein operator, which is an operator
    variational objective useful for SVGD.

    BROKEN -- FIX!?
    """
    def logp(y):
        return torch.log(p(y))
    nabla_logp = grad(logp(x), x, create_graph=True, grad_outputs=torch.ones_like(x))[0]
    phi_x = phi(x)
    return torch.mm(phi_x, nabla_logp) + grad(phi_x, x, create_graph=True)[0]

### Kernelized Stein discrepancy
class KSD:
    """
    Kernelized Stein discrepancy for a kernel in a
    given RKHS.
    """
    def __init__(self, kernel, p):
        self.kernel = kernel
        self.p = p

    def logp(self, x):
        return torch.log(self.p(x))
        
    def optimal_fn(self, particles):
        """
        Computes the optimal function for kernelized Stein
        discrepancy given kernel of RKHS.
        """
        pcls = differentiable(particles)
        def phi_star(x):
            K = self.kernel.eval(pcls, x)
            grad_logp = grad(self.logp(pcls), pcls, grad_outputs=torch.ones_like(self.logp(pcls)))[0]
            term1 = torch.mm(K, grad_logp)
            term2 = -torch.sum(grad(K, pcls, grad_outputs=torch.ones_like(K))[0], 1, keepdim=True)
            return (term1 + term2) / particles.shape[0]
    
        return phi_star
            
    def eval(self, q, num_samples=100):
        """
        Monte carlo estimate of KSD.
        """
        samples = q.sample((num_samples,))
        phi_star = self.optimal_fn(samples)
        # get new samples for second expectation
        new_samples = q.sample((num_samples,))
        ksd_vals = torch.zeros(samples.shape)
        for i in tqdm(range(num_samples)):
            sp = samples[i].clone().detach().requires_grad_()
            ksd_vals[i] = langevin_stein(self.p, phi_star, sp.view(-1))
        return torch.mean(ksd_vals)