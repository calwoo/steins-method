import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

from stein.operators import langevin_stein, KSD
from stein.utils import *

"""
Stein variational gradient descent from

[Q. Liu, D. Wang] Stein variational gradient descent: 
a general purpose bayesian inference algorithm
"""
class SVGD:
    def __init__(self, kernel, p):
        """
        INPUT:
            p = target distribution we wish to perform
                variational gradient descent towards
            kernel = kernel of the corresponding RKHS
        """
        self.p = p
        self.kernel = kernel
        # get KSD objective
        self.KSD = KSD(kernel, p)

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
        """
        new_particles = torch.zeros_like(self.particles)
        for i in range(self.num_particles):
            # get current particle
            particle = differentiable(self.particles[i])
            # compute gradient flow direction
            flow = self.phi(particle)
            # gradient descent!
            new_particles[i] = particle + lr * flow
        # update
        self.seed(new_particles)
        """
        flow = self.phi(self.particles)
        new_particles = self.particles + lr * flow
        self.seed(new_particles)

    def train(self, lr=1e-2, epochs=10):
        """
        Perform Stein variational gradient descent on
        set of initial particles.
        """
        for _ in tqdm(range(epochs)):
            self.step(lr=lr)