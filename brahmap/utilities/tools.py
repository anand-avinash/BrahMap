import numpy as np
import brahmap


def parallel_norm(x: np.ndarray):
    sqnorm = x.dot(x)
    sqnorm = brahmap.bMPI.comm.allreduce(sqnorm)
    ret = np.sqrt(sqnorm)
    return ret


class modify_numpy_context(object):
    def __init__(self):
        self.parallel_norm = parallel_norm
        self.original_norm = np.linalg.norm

    def __enter__(self):
        np.linalg.norm = self.parallel_norm

    def __exit__(self, exc_type, exc_val, exc_tb):
        np.linalg.norm = self.original_norm


class TypeChangeWarning(Warning):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)
