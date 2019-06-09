"""
Visualization functions. Rudimentary library.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import seaborn as sns
import time
import os
import imageio

import torch


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
Visualization functions.
"""
def plot_dist(p, particles, bins=30, save=False, idx=0):
    # convert to numpy
    pcls = particles.detach().numpy()
    sns.distplot(pcls, hist=True, kde=True, bins=bins, norm_hist=True)
    # plot density
    xs = np.linspace(-10,10,1000)

    ax = plt.gca()
    ax.set_xlim(-13,10)
    ax.set_ylim(0,0.4)
    plt.plot(xs, p(torch.tensor(xs)).detach().numpy())
    
    if save:
        # create folder to save images in
        if not os.path.exists("imgs"):
            os.mkdir("imgs")
        # save image
        plt.savefig("imgs/{}".format(idx))
        plt.clf()
    else:
        plt.show()



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

def create_gif(folder, filenames):
    images = []
    for filename in filenames:
        images.append(imageio.imread(folder + "/" + filename))
    imageio.mimsave(folder + "/gmm-svgd.gif", images, duration=0.3)
    