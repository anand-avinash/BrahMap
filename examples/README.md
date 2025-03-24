# `BrahMap` Examples

While `BrahMap` offers wrapper functions for GLS map-making, users can also
take advantage of its modularity to do map-making in different ways. In
addition, `BrahMap` can perform the map-making for any arbitrary sky pixel
scheme. In this directory, we are sharing the example scripts and notebooks to
illustrate map-making using both explicit and wrapper approach for both
rectangular maps and healpix maps.

In case of any questions or comments, please feel free to reach out to us or
open an [issue](https://github.com/anand-avinash/BrahMap/issues/new).

## Jupyter notebooks

- [`rectangular_I_map_explicit.ipynb`](rectangular_I_map_explicit.ipynb)  
  This example presents the GLS map-making for $I$ component over a
  rectangular map, using the linear operators explicitly. Its workflow can be
  modified to compute the maps in different ways.

- [`healpix_QU_map_wrapper.ipynb`](healpix_QU_map_wrapper.ipynb)  
  This notebook presents the GLS map-making for $Q$ and $U$ components over a
  healpix map, using the dedicated wrapper function for the same.

- [`lbsim_IQU_map_explicit.ipynb`](lbsim_IQU_map_explicit.ipynb)  
  This notebook presents the GLS map-making for $I$, $Q$ and $U$ components
  using a typical simulation from *LiteBIRD* simulation framework. The example
  uses the linear operators explicitly, and its workflow can be modified to
  compute the maps in different ways.

- [`lbsim_IQU_map_wrapper.ipynb`](lbsim_IQU_map_wrapper.ipynb)  
  This notebook presents the GLS map-making for $I$, $Q$ and $U$ components
  using a typical simulation from *LiteBIRD* simulation framework. The example
  uses the dedicated wrapper function for GLS map-making from `litebid_sim`
  simulations.

## Python scripts

The following scripts are meant to be executed with MPI. A typical usage would
be `mpirun -n <nprocs> python <filename>` where `nprocs` refers to the number
of MPI processes to be used and `filename` refers to the filename of the
script.

- [`rectangular_I_map_wrapper.py`](rectangular_I_map_wrapper.py)  
  This script illustrates the GLS map-making for $I$ component over a
  rectangular map with the data distributed across multiple MPI processes. The
  example uses the dedicated wrapper functions for GLS map-making.

- [`healpix_QU_map_explicit.py`](healpix_QU_map_explicit.py)  
  This script presents the GLS map-making for $Q$ and $U$ components over
  healpix maps with the data distributed across multiple MPI processes. The
  example uses the linear operators explicitly, and its workflow can be
  modified to compute the maps in different ways.

- [`lbsim_IQU_map_wrapper.py`](lbsim_IQU_map_wrapper.py)  
  The script presents the GLS map-making for $I$, $Q$ and $U$ components using
  a typical parallel simulation from *LiteBIRD* simulation framework. The
  example uses the dedicated wrapper function for GLS map-making from
  `litebid_sim` simulations.
