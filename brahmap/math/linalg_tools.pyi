"""linalg_tools"""

from typing import Any

from numpy.typing import NDArray

def multiply_array(
    nsamples: int, diag: NDArray[Any], vec: NDArray[Any], prod: NDArray[Any]
) -> None: ...
