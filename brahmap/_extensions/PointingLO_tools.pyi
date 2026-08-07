"""PointingLO_tools"""

from typing import Any

import numpy
from numpy.typing import NDArray

def PLO_mult_I(
    nsamples: int,
    pointings: NDArray[Any],
    pointings_flag: NDArray[numpy.bool_] | None,
    vec: NDArray[Any],
    prod: NDArray[Any],
) -> None: ...
def PLO_rmult_I(
    new_npix: int,
    nsamples: int,
    pointings: NDArray[Any],
    pointings_flag: NDArray[numpy.bool_] | None,
    vec: NDArray[Any],
    prod: NDArray[Any],
    comm: object,
) -> None: ...
def PLO_mult_QU(
    nsamples: int,
    pointings: NDArray[Any],
    pointings_flag: NDArray[numpy.bool_] | None,
    sin2phi: NDArray[Any],
    cos2phi: NDArray[Any],
    vec: NDArray[Any],
    prod: NDArray[Any],
) -> None: ...
def PLO_rmult_QU(
    new_npix: int,
    nsamples: int,
    pointings: NDArray[Any],
    pointings_flag: NDArray[numpy.bool_] | None,
    sin2phi: NDArray[Any],
    cos2phi: NDArray[Any],
    vec: NDArray[Any],
    prod: NDArray[Any],
    comm: object,
) -> None: ...
def PLO_mult_IQU(
    nsamples: int,
    pointings: NDArray[Any],
    pointings_flag: NDArray[numpy.bool_] | None,
    sin2phi: NDArray[Any],
    cos2phi: NDArray[Any],
    vec: NDArray[Any],
    prod: NDArray[Any],
) -> None: ...
def PLO_rmult_IQU(
    new_npix: int,
    nsamples: int,
    pointings: NDArray[Any],
    pointings_flag: NDArray[numpy.bool_] | None,
    sin2phi: NDArray[Any],
    cos2phi: NDArray[Any],
    vec: NDArray[Any],
    prod: NDArray[Any],
    comm: object,
) -> None: ...
def shmem_PLO_rmult_I(
    new_npix: int,
    nsamples: int,
    pointings: NDArray[Any],
    pointings_flag: NDArray[numpy.bool_] | None,
    vec: NDArray[Any],
    grp_prod: NDArray[Any],
    win_grp_prod: object,
    node_prod: NDArray[Any],
    win_node_prod: object,
    node_root: int,
    tree_grp_comm: object,
    tree_grp_root_comm: object,
    node_comm: object,
    node_root_comm: object,
) -> None: ...
def shmem_PLO_rmult_QU(
    new_npix: int,
    nsamples: int,
    pointings: NDArray[Any],
    pointings_flag: NDArray[numpy.bool_] | None,
    sin2phi: NDArray[Any],
    cos2phi: NDArray[Any],
    vec: NDArray[Any],
    grp_prod: NDArray[Any],
    win_grp_prod: object,
    node_prod: NDArray[Any],
    win_node_prod: object,
    node_root: int,
    tree_grp_comm: object,
    tree_grp_root_comm: object,
    node_comm: object,
    node_root_comm: object,
) -> None: ...
def shmem_PLO_rmult_IQU(
    new_npix: int,
    nsamples: int,
    pointings: NDArray[Any],
    pointings_flag: NDArray[numpy.bool_] | None,
    sin2phi: NDArray[Any],
    cos2phi: NDArray[Any],
    vec: NDArray[Any],
    grp_prod: NDArray[Any],
    win_grp_prod: object,
    node_prod: NDArray[Any],
    win_node_prod: object,
    node_root: int,
    tree_grp_comm: object,
    tree_grp_root_comm: object,
    node_comm: object,
    node_root_comm: object,
) -> None: ...
