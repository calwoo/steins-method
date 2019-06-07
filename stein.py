### imports
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


### langevin-stein operator
def langevin_stein(p, phi, x):
    """
    implementation of the Langevin-Stein operator, which is an operator
    variational objective useful for SVGD.
    """
    # attractive term
    x_a = x.detach().requires_grad_()
    log_prob = torch.log(p(x_a))
    log_prob.backward()
    a_term = torch.dot(phi(x_a).view(-1), x_a.grad)
    
    # repulsive term
    x_r = x.detach().requires_grad_()
    phi_x = phi(x_r)
    phi_x.backward(torch.ones_like(x_r))
    r_term = torch.sum(x_r.grad)
    
    return a_term + r_term

### Kernelized Stein discrepancy
class KSD:
    """
    Kernelized Stein discrepancy for a kernel in a
    given RKHS.
    """
    def __init__(self, kernel, p):
        self.kernel = kernel
        self.p = p
        
    def optimal_fn(self, q, samples):
        # python lacks true currying, so this is
        # going to be sorta awkward
        def kern_curry(y):
            def k(x):
                return self.kernel.eval(x,y).view(-1)
            return k
        
        def phi(x):
            num_samples = samples.shape[0]
            phi_vals = torch.zeros(samples.shape)
            for i in range(num_samples):
                sp = samples[i]
                phi_vals[i] = langevin_stein(self.p, kern_curry(sp), x.view(-1))
            return torch.mean(phi_vals)
    
        return phi
            
    def eval(self, q, num_samples=100):
        """
        Monte carlo estimate of KSD.
        """
        samples = q.sample((num_samples,))
        phi = self.optimal_fn(q, samples)
        # get new samples for second expectation
        new_samples = q.sample((num_samples,))
        ksd_vals = torch.zeros(samples.shape)
        for i in tqdm(range(num_samples)):
            sp = new_samples[i]
            ksd_vals[i] = langevin_stein(self.p, phi, sp.view(-1))
        return torch.mean(ksd_vals)