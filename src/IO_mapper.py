import tensorflow as tf
import math

"""
(x - min_x) / (max_x - min_x) = (y - min_y) / (max_y - min_y)
"""
def linear_map(x, min_x, max_x, min_y, max_y):
    y = (x - min_x) / (max_x - min_x) * (max_y - min_y) + min_y
    return y


class IOMapper(object):
    def __init__(self):
        raise NotImplementedError("")

    def map_input(self, x):
        raise NotImplementedError("")

    def map_texture(self, x):
        raise NotImplementedError("")

    def map_output(self, x):
        raise NotImplementedError("")


class IdentityIOMapper(IOMapper):
    def __init__(self):
        pass

    def map_input(self, x):
        return x

    def map_texture(self, x):
        return x

    def map_output(self, x):
        return x

class LogIOMapper(IOMapper):
    def __init__(self, data_max_val):
        self.M = data_max_val

    """
    [0, M] => [-1, 1]
    """
    def map_input(self, x):
        y = linear_map(x, 0.0, self.M, 1.0/math.e, self.M + 1.0/math.e)  # [0, M] => [1/e, M + 1/e]
        y = tf.log(y) # => [-1, log(M + 1/e)]
        y = linear_map(y, -1.0, tf.log(self.M + 1.0 / math.e), -1.0, 1.0) # => [-1, 1]
        return y

    """
    [-inf, inf] => [-1, inf]
    And map 0 ==> 0
    """
    def map_texture(self, x):
        y = x + 1.0 - 1.0/math.e  # shift zero
        y = tf.maximum(x, tf.zeros_like(y)) + 1.0 / math.e # => [1/e, inf]
        y = tf.log(y) #[-1, inf]
        return y


    """
    [-1,1] => [0, M]
    """
    def map_output(self, x):
        y = linear_map(x, -1.0, 1.0, -1.0, tf.log(self.M + 1.0 / math.e)) # => [-1, log(M + 1/e)]
        y = tf.exp(y) # => [1/e, M + 1/e]
        y = linear_map(y, 1.0/math.e, self.M + 1.0/math.e, 0.0, self.M) # => [0, M]
        return y



