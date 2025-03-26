# Installation

The versions of the dependencies for `BrahMap` are quite flexible. If you
intend to use `BrahMap` alongside other packages (like `litebird_sim`), we
recommend installing `BrahMap` after you have installed those packages in the
same environment.

`BrahMap` requires an MPI library to compile the C++ extensions. By default,
it uses the `mpicxx` compiler wrapper for this task. However, you can override
this default by setting the `MPICXX` environment variable to your preferred
compiler. To install `BrahMap`, please follow these steps:

```bash
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

!!! note
    `BrahMap` uses [Setuptools](https://setuptools.pypa.io/en/latest/index.html)
    to build the C++ extensions. By default, it generates the compilation
    command by gathering various environment variables. You can customize the
    compilation flags used during the installation by setting the `CXXFLAGS`,
    `CPPFLAGS`, and `LDFLAGS` environment variables.
