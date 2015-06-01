__author__ = 'Alex'
import numpy

"""
****************************************************************
                    UTILITY FUNCTIONS
****************************************************************
"""

def mscb(t):
    """
    Find the index of the most significant change bit,
    the bit that will change when t is incremented
    """
    return int(numpy.log2(t ^ (t + 1)))

def log_sum_exp(v):
    """
    Calculate the log of the sum of exponentials of the vector elements.

    Use the shifting trick to minimize numerical precision errors.
    Argument is a list-like thing
    """
    number_actions = max(v)
    x = number_actions * numpy.ones(numpy.size(v))
    return number_actions + numpy.log(sum(numpy.exp(v - x)))
