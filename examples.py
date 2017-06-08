#!/usr/bin/env python
import pylab as plt

import gradients
from gradients.plottools import *

def example_all():
    """
    Run all examples.
    """
    plt.ion()
    plot_logmixture_binormal()
    plot_mixture_binormal(title='example_mixture_binormal')
    plot_logbinormal(cov = [[1.0, 0.5], [0.5, 1.0]],
                     title='example_logbinormal_corr')
    plot_binormal(cov = [[1.0, 0.5], [0.5, 1.0]],
                  title='example_binormal_corr')
    plot_logbinormal(title='example_logbinormal_uncorr')
    plot_binormal(title='example_binormal_uncorr')
    return

if __name__ == '__main__':
    example_all()

