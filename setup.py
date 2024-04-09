import os
from setuptools import Extension, setup

# g++ -O3 -march=native -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) example9.cpp -o example9$(python3-config --extension-suffix)

ext1 = Extension(
    "compute_weights",
    language="c++",
    sources=[os.path.join("brahmap", "src", "compute_weights.cpp")],
    include_dirs=[
        os.path.join("brahmap", "src"),
        os.path.join("extern", "pybind11", "include"),
    ],
    define_macros=None,
    extra_compile_args=[
        "-O3",
        # "-march=native",
        "-Wall",
        "-shared",
        "-std=c++20",
        "-fPIC",
        "-fvisibility=hidden",
        "-lm",
    ],
)

ext2 = Extension(
    "repixelize",
    sources=[os.path.join("brahmap", "src", "repixelization.cpp")],
    include_dirs=[
        os.path.join("brahmap", "src"),
        os.path.join("extern", "pybind11", "include"),
    ],
    define_macros=None,
    extra_compile_args=[
        "-O3",
        # "-march=native",
        "-Wall",
        "-shared",
        "-std=c++20",
        "-fPIC",
        "-fvisibility=hidden",
    ],
)

ext3 = Extension(
    "PointingLO_tools",
    sources=[os.path.join("brahmap", "src", "PointingLO_tools.cpp")],
    include_dirs=[
        os.path.join("brahmap", "src"),
        os.path.join("extern", "pybind11", "include"),
    ],
    define_macros=None,
    extra_compile_args=[
        "-O3",
        # "-march=native",
        "-Wall",
        "-shared",
        "-std=c++20",
        "-fPIC",
        "-fvisibility=hidden",
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
        # "-march=native",
        "-Wall",
        "-shared",
        "-std=c++20",
        "-fPIC",
        "-fvisibility=hidden",
    ],
)

ext5 = Extension(
    "InvNoiseCov_tools",
    sources=[os.path.join("brahmap", "src", "InvNoiseCov_tools.cpp")],
    include_dirs=[
        os.path.join("brahmap", "src"),
        os.path.join("extern", "pybind11", "include"),
    ],
    define_macros=None,
    extra_compile_args=[
        "-O3",
        # "-march=native",
        "-Wall",
        "-shared",
        "-std=c++20",
        "-fPIC",
        "-fvisibility=hidden",
    ],
)

setup(
    ext_modules=[ext1, ext2, ext3, ext4, ext5],
    # include_package_data=True,
)
