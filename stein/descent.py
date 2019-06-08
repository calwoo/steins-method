import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

from stein.operators import langevin_stein, KSD


"""
Stein variational gradient descent from

[Q. Liu, D. Wang] Stein variational gradient descent: 
a general purpose bayesian inference algorithm
"""
class SVGD:
    def __init__(self, p, kernel):
        """
        INPUT:
            p = target distribution we wish to perform
                variational gradient descent towards
            kernel = kernel of the corresponding RKHS
        """
        self.p = p
        self.kernel = kernel
        # get KSD objective
        self.KSD = KSD(p, kernel)

    def seed(self, particles):
        """
        set of initial particles (tensor of shape (num_particles, dim)) 
        that will be manipulated during the descent.
        """
        self.particles = particles
        self.num_particles = particles.shape[0]
        # optimal gradient flow
        self.phi = self.KSD.optimal_fn(particles)

    def step(self, lr=1e-2):
        return lr