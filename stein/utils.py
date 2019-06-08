"""
Visualization functions. Rudimentary library.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import seaborn as sns

import torch


"""
Visualization functions.
"""
def plot_dist(particles, bins=30):
    # convert to numpy
    pcls = particles.detach().numpy()
    sns.distplot(pcls, hist=True, kde=True, bins=bins)
    plt.show()



"""
General tensor transformations.
"""
def differentiable(tensor):
    """
    Set a tensor to a differentiable state, isolated
    from original tensor.
    """
    return tensor.clone().detach().requires_grad_()