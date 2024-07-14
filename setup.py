import os
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import mpi4py
import warnings
import subprocess

# g++ -O3 -march=native -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) example9.cpp -o example9$(python3-config --extension-suffix)

#############################
### compiler independent args
#############################
compiler_args = [
    "-pthread",
    "-O3",
    "-Wall",
    "-fPIC",
    "-fwrapv",
    "-fvisibility=hidden",
    "-std=c++20",
]

# These options are common with `compiler_so_args`. And since I supplying these options to `extra_link_args` of `Extension`, it will appear twice in the executable.
linker_so_args = ["-pthread", "-shared"]


##################################
### args that depends on compilers
##################################

# Intel compilers
intel_compile_args = ["-qopenmp", "-march=core-avx2"]
intel_link_args = ["-qopenmp"]

# GCC compilers
gcc_compile_args = ["-fopenmp", "-march=native"]
gcc_link_args = ["-fopenmp"]

# CLANG compilers
clang_compile_args = ["-fopenmp"]
clang_link_args = ["-fopenmp"]


### `compiler_so_args` is meant to be used in `compiler_so` for the linking phase. As of now, it is no different from the one used in `compiler_cxx`
compiler_so_args = compiler_args
linker_exe_args = linker_so_args


class brahmap_build_ext(build_ext):
    def get_environ_vars(self):
        if "CXX" in os.environ:
            CXX = os.environ["CXX"]
        else:
            CXX = "mpicxx"

        cxxflags = []
        if "CXXFLAGS" in os.environ:
            cxxflags.append(os.environ["CXXFLAGS"])

        cppflags = []
        if "CPPFLAGS" in os.environ:
            cppflags.append(os.environ["CPPFLAGS"])

        return CXX, cxxflags, cppflags

    def get_compiler_specific_flags(self, CXX):
        compiler_flags = []
        linker_flags = []
        try:
            result = subprocess.run([CXX, "--version"], capture_output=True, text=True)
            output_txt = result.stdout.lower()

            if "nvidia" in output_txt:
                pass
            elif "intel" in output_txt:
                compiler_flags = intel_compile_args
                linker_flags = intel_link_args
            elif "clang" in output_txt:
                compiler_flags = clang_compile_args
                linker_flags = clang_link_args
            elif "gcc" in output_txt:
                compiler_flags = gcc_compile_args
                linker_flags = gcc_link_args
            else:
                warnings.warn(
                    "Compiler not identified. Will proceed with default flags",
                    RuntimeWarning,
                )
        except Exception as e:
            print(
                f"{e}: Unable to detect compiler type. Will proceed with the default configurations"
            )

        return compiler_flags, linker_flags

    def build_extensions(self) -> None:
        CXX, CXXFLAGS1, CPPFLAGS = self.get_environ_vars()
        CXXFLAGS2, linker_flag = self.get_compiler_specific_flags(CXX)

        self.compiler.set_executable(
            "compiler_so", [CXX] + CPPFLAGS + CXXFLAGS1 + CXXFLAGS2 + compiler_args
        )
        # The following is meant for C compilation, but keeping it for the sake of completeness
        self.compiler.set_executable(
            "compiler", [CXX] + CPPFLAGS + CXXFLAGS1 + CXXFLAGS2 + compiler_args
        )
        self.compiler.set_executable("compiler_cxx", self.compiler.compiler)
        # don't think the following two are being used, but will keep them for the sake of completeness
        self.compiler.set_executable("linker_so", [CXX] + linker_flag)
        self.compiler.set_executable("linker_exe", [CXX] + linker_flag)

        super().build_extensions()


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
    "brahmap._extensions.InvNoiseCov_tools",
    sources=[os.path.join("brahmap", "_extensions", "InvNoiseCov_tools.cpp")],
    include_dirs=[
        os.path.join("brahmap", "_extensions"),
        os.path.join("extern", "pybind11", "include"),
    ],
    define_macros=None,
    extra_link_args=linker_so_args,
)

setup(
    ext_modules=[ext1, ext2, ext3, ext4, ext5],
    cmdclass={"build_ext": brahmap_build_ext},
    # include_package_data=True,
)
