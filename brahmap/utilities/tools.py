import cProfile
from typing import Any
import numpy as np
from ..math import parallel_norm


class bash_colors:
    """A utility class containing ANSI escape codes for printing colored
    text to the screen."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    def header(self, string: Any) -> str:
        return self.HEADER + str(string) + self.ENDC

    def blue(self, string: Any) -> str:
        return self.OKBLUE + str(string) + self.ENDC

    def green(self, string: Any) -> str:
        return self.OKGREEN + str(string) + self.ENDC

    def warning(self, string: Any) -> str:
        return self.WARNING + str(string) + self.ENDC

    def fail(self, string: Any) -> str:
        return self.FAIL + str(string) + self.ENDC

    def bold(self, string: Any) -> str:
        return self.BOLD + str(string) + self.ENDC

    def underline(self, string: Any) -> str:
        return self.UNDERLINE + str(string) + self.ENDC


class modify_numpy_context(object):
    """A context manager that temporarily replaces `np.linalg.norm` with
    `brahmap.math.parallel_norm`."""

    def __init__(self) -> None:
        self.parallel_norm = parallel_norm
        self.original_norm = np.linalg.norm

    def __enter__(self) -> None:
        setattr(np.linalg, "norm", self.parallel_norm)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        setattr(np.linalg, "norm", self.original_norm)


def profile_run() -> cProfile.Profile:
    """Profiles the execution of the code using the `cProfile` module.

    Returns
    -------
    cProfile.Profile
        The active profiling instance
    """
    pr = cProfile.Profile()
    return pr


def output_profile(pr: cProfile.Profile) -> None:
    """Outputs the collected profiling statistics from `profile_run`.

    Parameters
    ----------
    pr : cProfile.Profile
        A profiling or logging object used for performance tracking
    """
    import pstats
    import io

    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    pass
