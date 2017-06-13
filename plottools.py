import numpy as np
import pylab as plt
from math import sqrt
from . import core as gg

def example_all():
    """
    Run all examples.
    """
    plot_logmixture_binormal()
    plot_mixture_binormal(title='example_mixture_binormal')
    plot_logbinormal(cov = [[1.0, 0.5], [0.5, 1.0]],
                     title='example_logbinormal_corr')
    plot_binormal(cov = [[1.0, 0.5], [0.5, 1.0]],
                  title='example_binormal_corr')
    plot_logbinormal(title='example_logbinormal_uncorr')
    plot_binormal(title='example_binormal_uncorr')
    return
    
def plot_binormal(mean=[0.0, 0.0], cov=[[1., 0.],[0., 1.]], k=2,
                  title=''):
    """
    Plot gradient of binormal distribution.

    :param array-like mean: mean of distribution
    :param array-like cov: (2,2) covariance matrix
    :param string title: optional title for figure canvas
    :param float k: multiplicative factor defining extent of grid.
    """
    cov = np.array(cov)
    
    # Define coordinate grid
    x = np.linspace(mean[0] - k * sqrt(cov[0,0]),
                    mean[0] + k * sqrt(cov[0,0]), 15)
    y = np.linspace(mean[1] - k * sqrt(cov[1,1]),
                    mean[1] + k * sqrt(cov[1,1]), 15)
    xy = np.array(np.meshgrid(x, y)).T.reshape((-1, 2))
    
    # Compute gradient and norm
    grad = gg.gradient_normal(mean, cov, xy)
    gnorm = np.sqrt(np.sum(grad**2, axis=0))

    # Plot
    ax = prepare_axis(title)
    ax.quiver(xy[:,0], xy[:,1], grad[0], grad[1], gnorm, units='xy')
    ax.set_title('Binormal, mean={}, cov=[{},{}]'.format(mean, cov[0],
                                                         cov[1]),
                 fontsize=10)
    plt.show()
    return

def plot_logbinormal(mean=[0.0, 0.0], cov=[[1., 0.],[0., 1.]], k=2,
                     title=''):
    """
    Plot gradient of the logarithm of a binormal distribution.

    :param array-like mean: mean of distribution
    :param array-like cov: (2,2) covariance matrix
    :param string title: optional title for figure canvas
    :param float k: multiplicative factor defining extent of grid.
    """

    cov = np.array(cov)
    # Define coordinate grid
    x = np.linspace(mean[0] - k * sqrt(cov[0,0]),
                    mean[0] + k * sqrt(cov[0,0]), 15)
    y = np.linspace(mean[1] - k * sqrt(cov[1,1]),
                    mean[1] + k * sqrt(cov[1,1]), 15)
    xy = np.array(np.meshgrid(x, y)).T.reshape((-1, 2))

    # Compute gradient and norm
    grad = gg.gradient_lognormal(mean, cov, xy)
    gnorm = np.sqrt(np.sum(grad**2, axis=0))

    # Plot
    ax = prepare_axis(title)
    ax.quiver(xy[:,0], xy[:,1], grad[0], grad[1], gnorm, units='xy')
    ax.set_title('LogBinormal, mean={}, cov=[{},{}]'.format(mean, cov[0],
                                                         cov[1]),
                 fontsize=10)
    plt.show()
    return

def plot_mixture_binormal(mean1 = [0.0, 0.0], cov1 = [[1, 0.0],[0.0, 1]],
                          mean2 = [3.0, 0.0], cov2 = [[2, 0.8],[0.8, 2]],
                          alpha=0.5, title=''):
    """
    Plot gradient of a mixture of a binormal distributions.

    :param array-like mean1, mean2: mean of distributions
    :param array-like cov1, cov2: (2,2) covariance matrices
    :param float alpha: [0, 1) relative weight for first binormal.
    :param string title: optional title for figure canvas
    """
    cov1, cov2 = map(np.array, (cov1, cov2))
    mean1, mean2 = map(np.array, (mean1, mean2))

    # Define coordinate grid
    xmin = np.min([mean1[0], mean2[0]])
    ymin = np.min([mean1[1], mean2[1]])
    xmax = np.max([mean1[0], mean2[0]])
    ymax = np.max([mean1[1], mean2[1]])

    x = np.linspace(xmin - 2, xmax + 2, 30)
    y = np.linspace(ymin - 2, ymax +2, 15)
    xy = np.array(np.meshgrid(x, y)).T.reshape((-1, 2))

    # Compute gradient and norm
    grad = gg.gradient_mixturenormal(mean1, cov1, mean2, cov2, alpha, xy)
    gnorm = np.sqrt(np.sum(grad**2, axis=0))

    # Plot
    ax = prepare_axis(title)
    ax.quiver(xy[:,0], xy[:,1], grad[0], grad[1], gnorm, units='xy')
    ax.set_title('Mixture, '
                 'mean1={}, cov1=[{},{}]\n '
                 'mean2={}, cov2=[{},{}]'.format(mean1, cov1[0], cov1[1],
                                                 mean2, cov2[0], cov2[1]),
                 fontsize=10)
    plt.show()
    return

def plot_logmixture_binormal(mean1 = [0.0, 0.0], cov1 = [[1, 0.0],[0.0, 1]],
                             mean2 = [3.0, 0.0], cov2 = [[2, 0.8],[0.8, 2]],
                             alpha=0.5, title=''):
    """
    Plot gradient of a mixture of a binormal distributions.

    :param array-like mean1, mean2: mean of distributions
    :param array-like cov1, cov2: (2,2) covariance matrices
    :param float alpha: [0, 1) relative weight for first binormal.
    :param string title: optional title for figure canvas
    """
    cov1, cov2 = map(np.array, (cov1, cov2))
    mean1, mean2 = map(np.array, (mean1, mean2))

    # Define coordinate grid
    xmin = np.min([mean1[0], mean2[0]])
    ymin = np.min([mean1[1], mean2[1]])
    xmax = np.max([mean1[0], mean2[0]])
    ymax = np.max([mean1[1], mean2[1]])

    x = np.linspace(xmin - 2, xmax + 2, 30)
    y = np.linspace(ymin - 2, ymax +2, 15)
    xy = np.array(np.meshgrid(x, y)).T.reshape((-1, 2))

    # Compute gradient and norm
    grad = gg.gradient_logmixturenormal(mean1, cov1, mean2, cov2, 0.5, xy)
    gnorm = np.sqrt(np.sum(grad**2, axis=0))

    # Plot
    ax = prepare_axis('example_logmixture_binormal')
    ax.quiver(xy[:,0], xy[:,1], grad[0], grad[1], gnorm, units='xy')
    ax.set_title('Mixture, '
                 'mean1={}, cov1=[{},{}]\n '
                 'mean2={}, cov2=[{},{}]'.format(mean1, cov1[0], cov1[1],
                                                 mean2, cov2[0], cov2[1]),
                 fontsize=10)
    plt.show()
    return

def prepare_axis(title):
    fig = plt.figure()
    fig.canvas.set_window_title(title)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    return ax

