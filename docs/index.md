# Welcome to BrahMap

<!-- markdownlint-disable MD033 -->
<font color="red"> **This project is currently under active development!!!** </font>
<!-- markdownlint-disable MD033 -->

`BrahMap` is an optimal map-making framework for the future CMB experiments,
based on [COSMOMAP2](https://github.com/giuspugl/COSMOMAP2), as described in
[*Puglisi et al (2018)*](https://doi.org/10.1051/0004-6361/201832710).

`BrahMap` is written Python with C++ extension handling the heavy computation.
It implements GLS map-making with PCG solver, taking into account the
block-band diagonal noise correlation matrix. This implementation offers
solvers of I, QU, and IQU maps.

Go to the [quick start guide](quick_start.md) for a quick introduction to
map-making with `BrahMap`. Refer to the [API reference](api_references.md)
for a completer reference to `BrahMap` API.

## Installation

`BrahMap` can be installed with the following steps:

```shell
# Clone the repository
git clone --recursive https://github.com/anand-avinash/BrahMap.git

cd BrahMap

# Install the package
pip install .

# Alternatively, do an editable installation for development purpose
# followed by `pre-commit` install
pip install -e .
pre-commit install
```

### Notes

`BrahMap` uses [Setuptools](https://setuptools.pypa.io/en/latest/index.html)
to build the C++ extensions. By default, it creates the command for compilation
by collecting
[several environment variables](https://setuptools.pypa.io/en/latest/index.html).
To change the compiler for the building the C++ extension, supply the
compiler name via `CC` or `CXX` variables:

```shell
CC=clang++ pip install .
```
