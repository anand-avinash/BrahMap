# API reference

The `BrahMap` package features a three-level hierarchical abstraction to
separate low-level mathematical primitives from high-level experiment
interfaces:

- **Low-level interface**: Defines abstract base classes, basic linear
  operators, and the base pointing pre-processing container. It is
  represented by the `base` sub-module.
- **Mid-level interface**: Contains generic pointing operators, noise
  covariance models, preconditioners, and solvers - the core numerical
  components of the map-making pipelines. It builds upon the low-level
  foundation and is represented by the `core`, `math`, and `utilities`
  sub-modules.
- **High-level interface**: Acts as a bridge connecting the mid-level
  map-making pipeline to external simulation frameworks and databases.
  Currently, it features wrappers and integration tools for
  [`litebird_sim`](https://github.com/litebird/litebird_sim) under the
  `lbsim` sub-module.

Refer to the sections below for the detailed API reference of each component:

<!-- markdownlint-disable MD013 -->

## `litebird_sim` API

| Interface | Description |
| :--- | :--- |
| [`litebird_sim` Interface](./lbsim/index.md) | Wrappers and interface classes for integrating with `litebird_sim`. |

## Core API

| Interface | Description |
| :--- | :--- |
| [Core Map-making Interface](./core/index.md) | Core map-making operators, solvers, parameters, and result containers. |
| [Math Functions](./math_functions/index.md) | Numerical functions, linear algebra routines, and conjugate gradient solvers. |
| [Utilities](./utilities/index.md) | Parallelization managers, MPI helpers, and debugging utilities. |

## Base API

| Interface | Description |
| :--- | :--- |
| [Base Classes](./base_classes/index.md) | Abstract base classes for linear operators and pointing pre-processing. |
| [Miscellaneous](./misc/index.md) | Custom exceptions, warnings, and warning filtering tools. |

## C++ API

_Under development_ <!-- markdownlint-disable-line -->

<!-- markdownlint-enable MD013 -->
