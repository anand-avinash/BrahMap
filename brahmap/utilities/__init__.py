"""
This module contains the utilities functions strictly related to the computation, and input/output.

"""

from .utilities_functions import (
    is_sorted,
    bash_colors,
    filter_warnings,
    profile_run,
    output_profile,
    rescalepixels,
    angles_gen,
    pairs_gen,
    checking_output,
    noise_val,
    subscan_resize,
    system_setup,
)

from .tools import TypeChangeWarning

from .process_time_samples import ProcessTimeSamples, SolverType

from importlib.util import find_spec

# suggestion taken from: <https://docs.astral.sh/ruff/rules/unused-import/>
if find_spec("litebird_sim") is not None:
    from .lbsim_interface import lbs_process_timesamples

    __all__ = ["lbs_process_timesamples"]

else:
    __all__ = []

__all__ = __all__ + [
    "is_sorted",
    "bash_colors",
    "filter_warnings",
    "profile_run",
    "output_profile",
    "rescalepixels",
    "angles_gen",
    "pairs_gen",
    "checking_output",
    "noise_val",
    "subscan_resize",
    "system_setup",
    "ProcessTimeSamples",
    "SolverType",
    "TypeChangeWarning",
]
