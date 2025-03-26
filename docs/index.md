# Welcome to BrahMap

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

For a quick introduction to map-making with BrahMap, refer to the
[quick start guide](quick_start.md).
For a complete reference of the `BrahMap` API, refer to the
[API reference](api_reference.md). Complete example notebooks and scripts can
be [found here](https://github.com/anand-avinash/BrahMap/examples).

You can find detailed information on the implementation and features of
`BrahMap` at [arXiv:2501.16122](https://arxiv.org/abs/2501.16122).

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
