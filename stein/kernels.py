import numpy as np
import torch

from stein.utils import timed

class Kernel:
    """
    skeleton for a kernel
    """
    def eval(self, x, y):
        pass

class FeatureKernel(Kernel):
    """create kernel from feature map"""
    def __init__(self, feature_map=None):
        if feature_map is None:
            self.phi = lambda x: x
        else:
            self.phi = feature_map

    def eval(self, x, y):
        return torch.dot(phi(x), phi(y))


class PolynomialKernel(Kernel):
    """polynomial kernel (x.y + c)^d"""
    def __init__(self, degree, bias=0):
        self.d = degree
        self.c = bias
        # check if bias term is positive (to maintain positive-definiteness)
        assert self.c > 0

    def eval(self, x, y):
        dp = torch.dot(x, y)
        return (dp + self.c) ** self.d


class RBFKernel(Kernel):
    """radial basis (gaussian) kernel"""
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def eval(self, x, y):
        dist = torch.zeros((x.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                dist[i,j] = torch.sum((x[i]-y[j])**2)
        return torch.exp(- dist / (2 * self.sigma**2))

class SigmoidKernel(Kernel):
    def __init__(self, a=1, b=0):
        self.a = a
        self.b = b
        # check if terms are positive
        assert a >= 0 and b >= 0

    def eval(self, x, y):
        logit = a * torch.dot(x, y) + b
        return torch.tanh(logit)