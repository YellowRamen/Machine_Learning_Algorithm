#__author__ = 'Brian Huang'

import numpy.linalg as la
import numpy as np
from scipy.spatial.distance import euclidean

class Kernel(object):
    """Implements a list of kernels from
    http://en.wikipedia.org/wiki/Support_vector_machine
    """
    @staticmethod
    def linear():
        def f(x, y):
            return np.inner(x, y)
        return f

    @staticmethod
    def gaussian(sigma):
        def f(x, y):
            '''Courtesy to Ale for helping me fixing the square matrix error
            The Scipy package of Eculidean calculates the magnitude(norm)
            while taking in x and y and fixed the square matrix error. Originally,
            I wrote out the magnitude formula explicitly, which might have had
            manual errors.'''
            return np.exp(-(1./(2.*(sigma ** 2))) * (euclidean(x, y) ** 2))
        return f

    @staticmethod
    def _polykernel(dimension, offset):
        def f(x, y):
            return (np.inner(x,y) + offset) ** dimension
        return f

    @staticmethod
    def inhomogenous_polynomial(dimension):
        #x_i, x_j = dimension[0], dimension[1]
        def f(x, y):
            return (np.inner(x, y) + 1.) ** dimension
        return f

    @staticmethod
    def homogenous_polynomial(dimension):
        def f(x, y):
            return (np.inner(x, y) ** dimension)
        return f

    @staticmethod
    def hyperbolic_tangent(kappa, c):
        def f(x, y):
            return np.tanh(np.inner(kappa*x, y) + c)
        return f
