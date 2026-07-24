import pytest
import numpy as np
import brahmap
from brahmap import math


class InitParams:
    rng = np.random.default_rng(seed=[3234, brahmap.MPI_UTILS.rank])

    size = 128
    input_vec_f32 = rng.uniform(low=-1.0, high=1.0, size=size).astype(dtype=np.float32)
    input_vec_f64 = rng.uniform(low=-1.0, high=1.0, size=size).astype(dtype=np.float64)


class TestUnaryFunctions(InitParams):
    @pytest.mark.parametrize(
        "func_name",
        [
            "sin",
            "cos",
            "tan",
            "arcsin",
            "arccos",
            "arctan",
            "exp",
            "exp2",
            "log",
            "log2",
            "sqrt",
            "cbrt",
        ],
    )
    def test_unary_function(self, func_name):
        # Determine the input vectors
        if func_name in ["log", "log2", "sqrt"]:
            input_f32 = np.abs(self.input_vec_f32)
            input_f64 = np.abs(self.input_vec_f64)
        else:
            input_f32 = self.input_vec_f32
            input_f64 = self.input_vec_f64

        # Resolve function references
        brahmap_func = getattr(math, func_name)
        numpy_func = getattr(np, func_name)

        brahmap_vec_f32 = np.empty_like(input_f32)
        brahmap_vec_f64 = np.empty_like(input_f64)

        numpy_vec_f32 = np.empty_like(input_f32)
        numpy_vec_f64 = np.empty_like(input_f64)

        brahmap_func(self.size, input_f32, brahmap_vec_f32)
        numpy_func(input_f32, numpy_vec_f32)

        brahmap_func(self.size, input_f64, brahmap_vec_f64)
        numpy_func(input_f64, numpy_vec_f64)

        np.testing.assert_allclose(brahmap_vec_f32, numpy_vec_f32, rtol=1.5e-6)
        np.testing.assert_allclose(brahmap_vec_f64, numpy_vec_f64, rtol=1.5e-7)
