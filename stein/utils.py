"""
Visualization functions. Rudimentary library.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import seaborn as sns
import time

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



"""
Helper functions
"""
### time decorator
def timed(fn):
    def timed_fn(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        end = time.time()

        print("{} took {:2.2f} ms".format(fn.__name__, (end-start)*1000))
        return result
    return timed_fn