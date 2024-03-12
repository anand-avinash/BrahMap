import os
from setuptools import Extension, setup

# g++ -O3 -march=native -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) example9.cpp -o example9$(python3-config --extension-suffix)

ext1 = Extension(
    "process_samples",
    language="c++",
    sources=[os.path.join("brahmap", "src", "process_samples.cpp")],
    include_dirs=[
        os.path.join("brahmap", "src"),
        os.path.join("extern", "pybind11", "include"),
    ],
    define_macros=None,
    extra_compile_args=[
        "-O3",
        "-march=native",
        "-Wall",
        "-shared",
        "-std=c++14",
        "-fPIC",
        "-lm",
    ],
)

ext2 = Extension(
    "repixelize",
    sources=[os.path.join("brahmap", "src", "repixelize.cpp")],
    include_dirs=[
        os.path.join("brahmap", "src"),
        os.path.join("extern", "pybind11", "include"),
    ],
    define_macros=None,
    extra_compile_args=[
        "-O3",
        "-march=native",
        "-Wall",
        "-shared",
        "-std=c++14",
        "-fPIC",
    ],
)

ext3 = Extension(
    "SparseLO_tools",
    sources=[os.path.join("brahmap", "src", "SparseLO_tools.cpp")],
    include_dirs=[
        os.path.join("brahmap", "src"),
        os.path.join("extern", "pybind11", "include"),
    ],
    define_macros=None,
    extra_compile_args=[
        "-O3",
        "-march=native",
        "-Wall",
        "-shared",
        "-std=c++14",
        "-fPIC",
    ],
)

ext4 = Extension(
    "BlkDiagPrecondLO_tools",
    sources=[os.path.join("brahmap", "src", "BlkDiagPrecondLO_tools.cpp")],
    include_dirs=[
        os.path.join("brahmap", "src"),
        os.path.join("extern", "pybind11", "include"),
    ],
    define_macros=None,
    extra_compile_args=[
        "-O3",
        "-march=native",
        "-Wall",
        "-shared",
        "-std=c++14",
        "-fPIC",
    ],
)

setup(
    ext_modules=[ext1, ext2, ext3, ext4],
    # include_package_data=True,
)
