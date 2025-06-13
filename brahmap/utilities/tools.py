import warnings
import numpy as np
from ..math import parallel_norm


class bash_colors:
    """
    This class contains the necessary definitions to print to bash
    screen with colors. Sometimes it can be useful...
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    def header(self, string):
        return self.HEADER + str(string) + self.ENDC

    def blue(self, string):
        return self.OKBLUE + str(string) + self.ENDC

    def green(self, string):
        return self.OKGREEN + str(string) + self.ENDC

    def warning(self, string):
        return self.WARNING + str(string) + self.ENDC

    def fail(self, string):
        return self.FAIL + str(string) + self.ENDC

    def bold(self, string):
        return self.BOLD + str(string) + self.ENDC

    def underline(self, string):
        return self.UNDERLINE + str(string) + self.ENDC


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


class LowerTypeCastWarning(Warning):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


def filter_warnings(wfilter):
    """
    wfilter: {string}
    - "ignore": never print matching warnings;
    - "always": always print matching warnings

    """
    warnings.simplefilter(wfilter)


class ShapeError(Exception):
    """
    Exception class for handling shape mismatch errors.

    Exception raised when defining a linear operator of the wrong shape or
    multiplying a linear operator with a vector of the wrong shape.

    """

    def __init__(self, value):
        super(ShapeError, self).__init__()
        self.value = value

    def __str__(self):
        return repr(self.value)


def profile_run():
    """
    Profile the execution with :mod:`cProfile`
    """
    import cProfile

    pr = cProfile.Profile()
    return pr


def output_profile(pr):
    """
    Output of the profiling with :func:`profile_run`.

    **Parameter**

    - ``pr``:
        instance returned by :func:`profile_run`

    """
    import pstats
    import io

    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    pass
