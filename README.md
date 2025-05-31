<!-- markdownlint-disable MD033 -->
<p align="center"> <!-- markdownlint-disable-line -->
  <h1 align="center">BrahMap</h1>

  <h3 align="center">
  A scalable and modular map-making framework for the CMB experiments
  </h3>
  
  <p align="center">
  <strong>
      <a href="https://anand-avinash.github.io/BrahMap/">Documentation</a> |
      <a href="https://anand-avinash.github.io/BrahMap/quick_start/">
      Quick Start</a> |
      <a href="https://github.com/anand-avinash/BrahMap/tree/main/examples">
      Examples</a>
  </strong>
  </p>
</p>
<!-- markdownlint-enable MD033 -->

<!-- markdownlint-disable MD013 -->
![BrahMap testsuite](https://github.com/anand-avinash/BrahMap/actions/workflows/tests.yaml/badge.svg)
![BrahMap documentation build status](https://github.com/anand-avinash/BrahMap/actions/workflows/documentation.yaml/badge.svg)
<!-- markdownlint-enable MD013 -->

<!-- markdownlint-disable MD033 -->
<font color="red"> **This project is currently under active development!!!** </font>
<!-- markdownlint-enable MD033 -->

`BrahMap` is a scalable and modular map-making framework for the CMB
experiments. It features user-friendly Python interface for the linear
operators used in map-making. The Python interface simply handles the workflow
while delegating the heavy computations to the functions implemented in C++
extension. In addition to the interface for linear operators, `BrahMap` offers
a wrapper for Generalized Least Squares (GLS) map-making using the
Preconditioned Conjugate Gradient (PCG) solver. `BrahMap` is also integrated
with `litebird_sim` through dedicated wrappers.

For a quick introduction to map-making with `BrahMap`, refer to the
[quick start guide](https://anand-avinash.github.io/BrahMap/quick_start/).
For a complete reference of the `BrahMap` API, refer to the
[API reference](https://anand-avinash.github.io/BrahMap/api_reference/).
Complete example notebooks and scripts can be
[found here](./examples).

You can find detailed information on the implementation and features of
`BrahMap` at [arXiv:2501.16122](https://arxiv.org/abs/2501.16122).

## Installation

The versions of the dependencies for `BrahMap` are quite flexible. If you
intend to use `BrahMap` alongside other packages (like `litebird_sim`), we
recommend installing `BrahMap` after you have installed those packages in the
same environment.

`BrahMap` requires an MPI library to compile the C++ extensions. By default,
it uses the `mpicxx` compiler wrapper for this task. However, you can override
this default by setting the `MPICXX` environment variable to your preferred
compiler. To install `BrahMap`, please follow these steps:

```shell
# Clone the repository
git clone --recursive https://github.com/anand-avinash/BrahMap.git

# Enter the directory
cd BrahMap

# Set the compiler you want to use (optional)
export MPICXX=mpiicpc

# Install the package
pip install .

# Alternatively, do an editable installation for development purpose
# followed by `pre-commit` install
pip install -e .
pre-commit install
```

> [!NOTE]
> `BrahMap` uses [Setuptools](https://setuptools.pypa.io/en/latest/index.html)
> to build the C++ extensions. By default, it generates the compilation command
> by gathering various environment variables. You can customize the compilation
> flags used during the installation by setting the `CXXFLAGS`, `CPPFLAGS`,
> and `LDFLAGS` environment variables.

## Citation

This work can be cited with:

<!-- markdownlint-disable MD013 -->
```text
@misc{anand2025brahmap,
      title={\texttt{BrahMap}: A scalable and modular map-making framework for the CMB experiments}, 
      author={Avinash Anand and Giuseppe Puglisi},
      year={2025},
      eprint={2501.16122},
      archivePrefix={arXiv},
      primaryClass={astro-ph.CO},
      url={https://arxiv.org/abs/2501.16122}, 
}
```
<!-- markdownlint-enable MD013 -->

## Acknowledgement

This work is supported by Italian Research Center on High
Performance Computing, Big Data and Quantum Computing
(ICSC), project funded by European Union - NextGenerationEU - and National
Recovery and Resilience Plan (NRRP) - Mission 4 Component 2 within the
activities of Spoke 3 (Astrophysics and Cosmos Observations).
