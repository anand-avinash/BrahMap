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

- [`rectangular_I_map_explicit.ipynb`](rectangular_I_map_explicit.ipynb) [[open in Colab](https://colab.research.google.com/github/anand-avinash/BrahMap/blob/main/examples/rectangular_I_map_explicit.ipynb)]  <!-- markdownlint-disable MD013 -->  
  This example presents the GLS map-making for $I$ component over a
  rectangular map, using the linear operators explicitly. Its workflow can be
  modified to compute the maps in different ways.

- [`healpix_QU_map_wrapper.ipynb`](healpix_QU_map_wrapper.ipynb) [[open in Colab](https://colab.research.google.com/github/anand-avinash/BrahMap/blob/main/examples/healpix_QU_map_wrapper.ipynb)]  <!-- markdownlint-disable MD013 -->  
  This notebook presents the GLS map-making for $Q$ and $U$ components over a
  healpix map, using the dedicated wrapper function for the same.

- [`lbsim_IQU_map_explicit.ipynb`](lbsim_IQU_map_explicit.ipynb) [[open in Colab](https://colab.research.google.com/github/anand-avinash/BrahMap/blob/main/examples/lbsim_IQU_map_explicit.ipynb)]  <!-- markdownlint-disable MD013 -->  
  This notebook presents the GLS map-making for $I$, $Q$ and $U$ components
  using a typical simulation from *LiteBIRD* simulation framework. The example
  uses the linear operators explicitly, and its workflow can be modified to
  compute the maps in different ways.

- [`lbsim_IQU_map_wrapper.ipynb`](lbsim_IQU_map_wrapper.ipynb) [[open in Colab](https://colab.research.google.com/github/anand-avinash/BrahMap/blob/main/examples/lbsim_IQU_map_wrapper.ipynb)]  <!-- markdownlint-disable MD013 -->  
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

## Noise covariances and their inverse

In all the examples above, we have used white noise covariance for map-making.
BrahMap offers multiple types of noise covariances, all of which can replace
the white noise covariance used in these examples. The following example
notebooks provide an overview of the available noise covariances and
demonstrate various methods for creating them.

- [`basic_noise_covariances.ipynb`](basic_noise_covariances.ipynb) [[open in Colab](https://colab.research.google.com/github/anand-avinash/BrahMap/blob/main/examples/basic_noise_covariances.ipynb)]  <!-- markdownlint-disable MD013 -->  
  This notebook contains the examples related to the basic noise covariance
  operators.

- [`block_diagonal_noise_covariances.ipynb`](block_diagonal_noise_covariances.ipynb) [[open in Colab](https://colab.research.google.com/github/anand-avinash/BrahMap/blob/main/examples/block_diagonal_noise_covariances.ipynb)] <!-- markdownlint-disable MD013 -->  
  This notebook contains the examples for the block-diagonal noise covariances
  that corresponds to a noise time-stream made up of multiple stationary but
  mutually uncorrelated sections.

- [`lbsim_noise_covariances.ipynb`](lbsim_noise_covariances.ipynb) [[open in Colab](https://colab.research.google.com/github/anand-avinash/BrahMap/blob/main/examples/lbsim_noise_covariances.ipynb)]  <!-- markdownlint-disable MD013 -->  
  This notebook contains the examples for the noise covariance operators for
  the BrahMap's interface to `litebird_sim`.
