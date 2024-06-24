import os
from setuptools import Extension, setup
import mpi4py

# g++ -O3 -march=native -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) example9.cpp -o example9$(python3-config --extension-suffix)

compiler_args = [
    "-O3",
    "-Wall",
    "-shared",
    "-std=c++20",
    "-fPIC",
    "-fvisibility=hidden",
]


ext1 = Extension(
    "brahmap._extensions.compute_weights",
    language="c++",
    sources=[os.path.join("brahmap", "_extensions", "compute_weights.cpp")],
    include_dirs=[
        os.path.join("brahmap", "_extensions"),
        os.path.join("extern", "pybind11", "include"),
        os.path.join(mpi4py.get_include()),
    ],
    define_macros=None,
    extra_compile_args=compiler_args,
)

ext2 = Extension(
    "brahmap._extensions.repixelize",
    sources=[os.path.join("brahmap", "_extensions", "repixelization.cpp")],
    include_dirs=[
        os.path.join("brahmap", "_extensions"),
        os.path.join("extern", "pybind11", "include"),
    ],
    define_macros=None,
    extra_compile_args=compiler_args,
)

ext3 = Extension(
    "brahmap._extensions.PointingLO_tools",
    sources=[os.path.join("brahmap", "_extensions", "PointingLO_tools.cpp")],
    include_dirs=[
        os.path.join("brahmap", "_extensions"),
        os.path.join("extern", "pybind11", "include"),
        os.path.join(mpi4py.get_include()),
    ],
    define_macros=None,
    extra_compile_args=compiler_args,
)

ext4 = Extension(
    "brahmap._extensions.BlkDiagPrecondLO_tools",
    sources=[os.path.join("brahmap", "_extensions", "BlkDiagPrecondLO_tools.cpp")],
    include_dirs=[
        os.path.join("brahmap", "_extensions"),
        os.path.join("extern", "pybind11", "include"),
    ],
    define_macros=None,
    extra_compile_args=compiler_args,
)

ext5 = Extension(
    "brahmap._extensions.InvNoiseCov_tools",
    sources=[os.path.join("brahmap", "_extensions", "InvNoiseCov_tools.cpp")],
    include_dirs=[
        os.path.join("brahmap", "_extensions"),
        os.path.join("extern", "pybind11", "include"),
    ],
    define_macros=None,
    extra_compile_args=compiler_args,
)

setup(
    ext_modules=[ext1, ext2, ext3, ext4, ext5],
    # include_package_data=True,
)
