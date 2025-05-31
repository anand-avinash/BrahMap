import os
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools._distutils.ccompiler import new_compiler
import mpi4py
import threading
from typing import Any, Iterator
import tempfile
import shutil
from pathlib import Path
import contextlib


# g++ -O3 -march=native -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 \
# --includes) example9.cpp -o example9$(python3-config --extension-suffix)

#################################
### compiler independent args ###
#################################
compiler_args = [
    "-pthread",
    "-O3",
    "-Wall",
    "-fPIC",
    "-fwrapv",
    "-fvisibility=hidden",
    "-std=c++20",
]

# These options are common with `compiler_so_args`. And since I am supplying
# these options to `extra_link_args` of `Extension`, it will appear twice in
# the executable.
linker_so_args = ["-pthread"]


######################################
### args that depends on compilers ###
######################################

# OpenMP compilation flags
openmp_flags = [
    "-qopenmp",
    "-fopenmp",
    "-D_DISABLE_OMP",  # If none of the valid flags work, then disable OpenMP
]

# Performance tuning flags
performance_flags = [
    "-march=native",
    "-mtune=native",
    "-mcpu=native",
    "",  # If none of the valid flags work, then use empty string
]


##################
### other args ###
##################

# `compiler_so_args` is meant to be used in `compiler_so` for the linking
# phase. As of now, it is no different from the one used in `compiler_cxx`
compiler_so_args = compiler_args
linker_exe_args = linker_so_args


##########################################################
### framework to check if a compiler flag is supported ###
##########################################################

# Adopted from
# <https://github.com/pybind/pybind11/blob/
# d2e7e8c68711d1ebfb02e2f20bd1cb3bfc5647c0/
# pybind11/setup_helpers.py#L89>


tmp_chdir_lock = threading.Lock()


@contextlib.contextmanager
def tmp_chdir() -> Iterator[str]:
    "Prepare and enter a temporary directory, cleanup when done"
    # Threadsafe
    with tmp_chdir_lock:
        olddir = os.getcwd()
        try:
            tmpdir = tempfile.mkdtemp()
            os.chdir(tmpdir)
            yield tmpdir
        finally:
            os.chdir(olddir)
            shutil.rmtree(tmpdir)


# cf http://bugs.python.org/issue26689
def check_flag(compiler: Any, flag: str) -> bool:
    """
    Return the flag if a flag name is supported on the
    specified compiler, otherwise None (can be used as a boolean).
    If multiple flags are passed, return the first that matches.
    """
    with tmp_chdir():
        fname = Path("flagcheck.cpp")
        # Don't trigger -Wunused-parameter.
        fname.write_text("int main (int, char **) { return 0; }", encoding="utf-8")

        try:
            compiler.compile([str(fname)], extra_postargs=[flag])
            return flag
        except Exception:
            return None


##############################################
### defining the dedicated build extension ###
##############################################


class brahmap_build_ext(build_ext):
    def get_environ_vars(self):
        if "MPICXX" in os.environ:
            MPICXX = os.environ["MPICXX"]
        else:
            MPICXX = "mpicxx"

        cxxflags = []
        if "CXXFLAGS" in os.environ:
            cxxflags.append(os.environ["CXXFLAGS"])

        cppflags = []
        if "CPPFLAGS" in os.environ:
            cppflags.append(os.environ["CPPFLAGS"])

        ldflags = []
        if "LDFLAGS" in os.environ:
            ldflags.append(os.environ["LDFLAGS"])

        if "BRAHMAP_DISABLE_OMP" in os.environ:
            cxxflags.append("-D_DISABLE_OMP")

        return MPICXX, cxxflags, cppflags, ldflags

    def get_compiler_specific_flags(self, MPICXX):
        compiler_flags = []
        linker_flags = []

        custom_compiler = new_compiler()
        custom_compiler.compiler = [MPICXX]
        custom_compiler.compiler_cxx = [MPICXX]

        for item in openmp_flags:
            if check_flag(custom_compiler, item) is not None:
                compiler_flags += [item]
                break

        for item in performance_flags:
            if check_flag(custom_compiler, item) is not None:
                compiler_flags += [item]
                break

        return compiler_flags, linker_flags

    def build_extensions(self) -> None:
        MPICXX, CXXFLAGS1, CPPFLAGS, LDFLAGS = self.get_environ_vars()
        CXXFLAGS2, linker_flags = self.get_compiler_specific_flags(MPICXX)

        # Producing the shared objects
        self.compiler.set_executable(
            "compiler_so",
            [MPICXX]
            + CPPFLAGS
            + CXXFLAGS1
            + CXXFLAGS2
            + compiler_so_args
            + linker_flags
            + LDFLAGS,
        )
        self.compiler.set_executable("compiler_so_cxx", self.compiler.compiler_so)

        # The following is meant for C compilation, but keeping it for the
        # sake of completeness
        self.compiler.set_executable(
            "compiler", [MPICXX] + CPPFLAGS + CXXFLAGS1 + CXXFLAGS2 + compiler_args
        )

        # Compilation
        self.compiler.set_executable("compiler_cxx", self.compiler.compiler)

        # I don't think the following two are being used, but will keep them
        # for the sake of completeness
        self.compiler.set_executable("linker_so", [MPICXX] + linker_flags + LDFLAGS)
        self.compiler.set_executable("linker_exe", [MPICXX] + linker_flags + LDFLAGS)

        super().build_extensions()


ext1 = Extension(
    "brahmap._extensions.compute_weights",
    sources=[os.path.join("brahmap", "_extensions", "compute_weights.cpp")],
    include_dirs=[
        os.path.join("brahmap", "_extensions"),
        os.path.join("extern", "pybind11", "include"),
        os.path.join(mpi4py.get_include()),
    ],
    define_macros=None,
    extra_link_args=linker_so_args,
)

ext2 = Extension(
    "brahmap._extensions.repixelize",
    sources=[os.path.join("brahmap", "_extensions", "repixelization.cpp")],
    include_dirs=[
        os.path.join("brahmap", "_extensions"),
        os.path.join("extern", "pybind11", "include"),
    ],
    define_macros=None,
    extra_link_args=linker_so_args,
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
    extra_link_args=linker_so_args,
)

ext4 = Extension(
    "brahmap._extensions.BlkDiagPrecondLO_tools",
    sources=[os.path.join("brahmap", "_extensions", "BlkDiagPrecondLO_tools.cpp")],
    include_dirs=[
        os.path.join("brahmap", "_extensions"),
        os.path.join("extern", "pybind11", "include"),
    ],
    define_macros=None,
    extra_link_args=linker_so_args,
)

ext5 = Extension(
    "brahmap.math.linalg_tools",
    sources=[os.path.join("brahmap", "math", "linalg_tools.cpp")],
    include_dirs=[
        os.path.join("extern", "pybind11", "include"),
    ],
    define_macros=None,
    extra_link_args=linker_so_args,
)

ext6 = Extension(
    "brahmap.math.unary_functions",
    sources=[os.path.join("brahmap", "math", "unary_functions.cpp")],
    include_dirs=[
        os.path.join("extern", "pybind11", "include"),
    ],
    define_macros=None,
    extra_link_args=linker_so_args,
)

setup(
    ext_modules=[ext1, ext2, ext3, ext4, ext5, ext6],
    cmdclass={"build_ext": brahmap_build_ext},
    # include_package_data=True,
)
