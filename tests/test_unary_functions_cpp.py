import pytest
import numpy as np
import brahmap
from brahmap import math

brahmap.Initialize()


class InitParams:
    rng = np.random.default_rng(seed=[3234, brahmap.bMPI.rank])

    size = 128
    input_vec_f32 = rng.uniform(low=-1.0, high=1.0, size=size).astype(dtype=np.float32)
    input_vec_f64 = rng.uniform(low=-1.0, high=1.0, size=size).astype(dtype=np.float64)

    brahmap_vec_f32 = np.empty_like(input_vec_f32)
    brahmap_vec_f64 = np.empty_like(input_vec_f64)

    numpy_vec_f32 = np.empty_like(input_vec_f32)
    numpy_vec_f64 = np.empty_like(input_vec_f64)


class TestUnaryFunctions(InitParams):
    def test_sin(self):
        math.sin(self.size, self.input_vec_f32, self.brahmap_vec_f32)
        np.sin(self.input_vec_f32, self.numpy_vec_f32)

        math.sin(self.size, self.input_vec_f64, self.brahmap_vec_f64)
        np.sin(self.input_vec_f64, self.numpy_vec_f64)

        np.testing.assert_allclose(
            self.brahmap_vec_f32, self.numpy_vec_f32, rtol=1.5e-6
        )
        np.testing.assert_allclose(
            self.brahmap_vec_f64, self.numpy_vec_f64, rtol=1.5e-7
        )

    def test_cos(self):
        math.cos(self.size, self.input_vec_f32, self.brahmap_vec_f32)
        np.cos(self.input_vec_f32, self.numpy_vec_f32)

        math.cos(self.size, self.input_vec_f64, self.brahmap_vec_f64)
        np.cos(self.input_vec_f64, self.numpy_vec_f64)

        np.testing.assert_allclose(
            self.brahmap_vec_f32, self.numpy_vec_f32, rtol=1.5e-6
        )
        np.testing.assert_allclose(
            self.brahmap_vec_f64, self.numpy_vec_f64, rtol=1.5e-7
        )

    def test_tan(self):
        math.tan(self.size, self.input_vec_f32, self.brahmap_vec_f32)
        np.tan(self.input_vec_f32, self.numpy_vec_f32)

        math.tan(self.size, self.input_vec_f64, self.brahmap_vec_f64)
        np.tan(self.input_vec_f64, self.numpy_vec_f64)

        np.testing.assert_allclose(
            self.brahmap_vec_f32, self.numpy_vec_f32, rtol=1.5e-6
        )
        np.testing.assert_allclose(
            self.brahmap_vec_f64, self.numpy_vec_f64, rtol=1.5e-7
        )

    def test_arcsin(self):
        math.arcsin(self.size, self.input_vec_f32, self.brahmap_vec_f32)
        np.arcsin(self.input_vec_f32, self.numpy_vec_f32)

        math.arcsin(self.size, self.input_vec_f64, self.brahmap_vec_f64)
        np.arcsin(self.input_vec_f64, self.numpy_vec_f64)

        np.testing.assert_allclose(
            self.brahmap_vec_f32, self.numpy_vec_f32, rtol=1.5e-6
        )
        np.testing.assert_allclose(
            self.brahmap_vec_f64, self.numpy_vec_f64, rtol=1.5e-7
        )

    def test_arccos(self):
        math.arccos(self.size, self.input_vec_f32, self.brahmap_vec_f32)
        np.arccos(self.input_vec_f32, self.numpy_vec_f32)

        math.arccos(self.size, self.input_vec_f64, self.brahmap_vec_f64)
        np.arccos(self.input_vec_f64, self.numpy_vec_f64)

        np.testing.assert_allclose(
            self.brahmap_vec_f32, self.numpy_vec_f32, rtol=1.5e-6
        )
        np.testing.assert_allclose(
            self.brahmap_vec_f64, self.numpy_vec_f64, rtol=1.5e-7
        )

    def test_arctan(self):
        math.arctan(self.size, self.input_vec_f32, self.brahmap_vec_f32)
        np.arctan(self.input_vec_f32, self.numpy_vec_f32)

        math.arctan(self.size, self.input_vec_f64, self.brahmap_vec_f64)
        np.arctan(self.input_vec_f64, self.numpy_vec_f64)

        np.testing.assert_allclose(
            self.brahmap_vec_f32, self.numpy_vec_f32, rtol=1.5e-6
        )
        np.testing.assert_allclose(
            self.brahmap_vec_f64, self.numpy_vec_f64, rtol=1.5e-7
        )

    def test_exp(self):
        math.exp(self.size, self.input_vec_f32, self.brahmap_vec_f32)
        np.exp(self.input_vec_f32, self.numpy_vec_f32)

        math.exp(self.size, self.input_vec_f64, self.brahmap_vec_f64)
        np.exp(self.input_vec_f64, self.numpy_vec_f64)

        np.testing.assert_allclose(
            self.brahmap_vec_f32, self.numpy_vec_f32, rtol=1.5e-6
        )
        np.testing.assert_allclose(
            self.brahmap_vec_f64, self.numpy_vec_f64, rtol=1.5e-7
        )

    def test_exp2(self):
        math.exp2(self.size, self.input_vec_f32, self.brahmap_vec_f32)
        np.exp2(self.input_vec_f32, self.numpy_vec_f32)

        math.exp2(self.size, self.input_vec_f64, self.brahmap_vec_f64)
        np.exp2(self.input_vec_f64, self.numpy_vec_f64)

        np.testing.assert_allclose(
            self.brahmap_vec_f32, self.numpy_vec_f32, rtol=1.5e-6
        )
        np.testing.assert_allclose(
            self.brahmap_vec_f64, self.numpy_vec_f64, rtol=1.5e-7
        )

    def test_log(self):
        math.log(self.size, np.abs(self.input_vec_f32), self.brahmap_vec_f32)
        np.log(np.abs(self.input_vec_f32), self.numpy_vec_f32)

        math.log(self.size, np.abs(self.input_vec_f64), self.brahmap_vec_f64)
        np.log(np.abs(self.input_vec_f64), self.numpy_vec_f64)

        np.testing.assert_allclose(
            self.brahmap_vec_f32, self.numpy_vec_f32, rtol=1.5e-6
        )
        np.testing.assert_allclose(
            self.brahmap_vec_f64, self.numpy_vec_f64, rtol=1.5e-7
        )

    def test_log2(self):
        math.log2(self.size, np.abs(self.input_vec_f32), self.brahmap_vec_f32)
        np.log2(np.abs(self.input_vec_f32), self.numpy_vec_f32)

        math.log2(self.size, np.abs(self.input_vec_f64), self.brahmap_vec_f64)
        np.log2(np.abs(self.input_vec_f64), self.numpy_vec_f64)

        np.testing.assert_allclose(
            self.brahmap_vec_f32, self.numpy_vec_f32, rtol=1.5e-6
        )
        np.testing.assert_allclose(
            self.brahmap_vec_f64, self.numpy_vec_f64, rtol=1.5e-7
        )

    def test_sqrt(self):
        math.sqrt(self.size, np.abs(self.input_vec_f32), self.brahmap_vec_f32)
        np.sqrt(np.abs(self.input_vec_f32), self.numpy_vec_f32)

        math.sqrt(self.size, np.abs(self.input_vec_f64), self.brahmap_vec_f64)
        np.sqrt(np.abs(self.input_vec_f64), self.numpy_vec_f64)

        np.testing.assert_allclose(
            self.brahmap_vec_f32, self.numpy_vec_f32, rtol=1.5e-6
        )
        np.testing.assert_allclose(
            self.brahmap_vec_f64, self.numpy_vec_f64, rtol=1.5e-7
        )

    def test_cbrt(self):
        math.cbrt(self.size, self.input_vec_f32, self.brahmap_vec_f32)
        np.cbrt(self.input_vec_f32, self.numpy_vec_f32)

        math.cbrt(self.size, self.input_vec_f64, self.brahmap_vec_f64)
        np.cbrt(self.input_vec_f64, self.numpy_vec_f64)

        np.testing.assert_allclose(
            self.brahmap_vec_f32, self.numpy_vec_f32, rtol=1.5e-6
        )
        np.testing.assert_allclose(
            self.brahmap_vec_f64, self.numpy_vec_f64, rtol=1.5e-7
        )


if __name__ == "__main__":
    pytest.main([f"{__file__}::TestUnaryFunctions::test_sin", "-v", "-s"])
    pytest.main([f"{__file__}::TestUnaryFunctions::test_cos", "-v", "-s"])
    pytest.main([f"{__file__}::TestUnaryFunctions::test_tan", "-v", "-s"])
    pytest.main([f"{__file__}::TestUnaryFunctions::test_arcsin", "-v", "-s"])
    pytest.main([f"{__file__}::TestUnaryFunctions::test_arccos", "-v", "-s"])
    pytest.main([f"{__file__}::TestUnaryFunctions::test_arctan", "-v", "-s"])
    pytest.main([f"{__file__}::TestUnaryFunctions::test_exp", "-v", "-s"])
    pytest.main([f"{__file__}::TestUnaryFunctions::test_exp2", "-v", "-s"])
    pytest.main([f"{__file__}::TestUnaryFunctions::test_log", "-v", "-s"])
    pytest.main([f"{__file__}::TestUnaryFunctions::test_log2", "-v", "-s"])
    pytest.main([f"{__file__}::TestUnaryFunctions::test_sqrt", "-v", "-s"])
    pytest.main([f"{__file__}::TestUnaryFunctions::test_cbrt", "-v", "-s"])
