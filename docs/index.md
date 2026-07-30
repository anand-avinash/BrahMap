# Welcome to BrahMap

<!-- markdownlint-disable MD013 -->
![BrahMap testsuite](https://github.com/anand-avinash/BrahMap/actions/workflows/tests.yaml/badge.svg)
![BrahMap documentation build status](https://github.com/anand-avinash/BrahMap/actions/workflows/documentation.yaml/badge.svg)
<!-- markdownlint-enable MD013 -->

`BrahMap` is a scalable and modular map-making framework for CMB experiments.
It features a user-friendly Python interface for the linear operators used in
map-making. The Python interface seamlessly handles the workflow while
delegating the heavy computations to highly optimized C++ extensions. In
addition to the core linear operators, `BrahMap` offers a wrapper for
Generalized Least Squares (GLS) map-making using a Preconditioned Conjugate
Gradient (PCG) solver. `BrahMap` is also fully integrated with `litebird_sim`
through dedicated wrappers.

<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD013 -->
<style>
.center-last-card > ul > li:last-child {
  grid-column: 1 / -1;
  text-align: center;
}
</style>

<div class="grid cards center-last-card" markdown="1">

- :material-download: **[Installation](overview/installation.md)** – How to
  install and set up `BrahMap` on your system
- :material-rocket-launch: **[Quick Start](quick_start/index.md)** – A quick
  introduction to map-making with `BrahMap`
- :material-notebook-multiple: **[Examples](https://github.com/anand-avinash/BrahMap/tree/main/examples)** – Complete example notebooks and scripts to get started
- :material-book-open-page-variant: **[API Reference](api_reference/index.md)**
  – A complete reference to the `BrahMap` API
- :material-speedometer: **[Benchmarking](https://github.com/anand-avinash/BrahMap/tree/main/benchmarks)**
  – A performance benchmarking suite for core numerical routines

</div>
<!-- markdownlint-enable MD013 -->
<!-- markdownlint-enable MD033 -->

For detailed information on the implementation and features of `BrahMap`,
please refer to the paper
[arXiv:2501.16122](https://arxiv.org/abs/2501.16122).

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
