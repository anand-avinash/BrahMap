import numpy as np
import process_samples
import helper_initweigths as iw


class TestInitializeWeights_float32:
    # Testing on random data: <https://numpy.org/doc/stable/reference/testing.html#tests-on-random-data>
    np.random.seed(414)
    oldnpix = 128
    nsamples = oldnpix * 10
    w = np.random.random(size=nsamples).astype(dtype=np.float32)
    pixs = np.random.randint(low=0, high=oldnpix, size=nsamples)
    badpixs = np.random.randint(low=0, high=nsamples, size=100)
    pixs[badpixs] = -1
    phi = np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=nsamples).astype(
        dtype=np.float32
    )

    def test_initializeweights_pol1(self):
        cpp_counts, cpp_mask = process_samples.py_process_pol1(
            self.nsamples, self.oldnpix, self.w, self.pixs
        )

        py_counts, py_mask = iw.initializeweights_pol1(
            self.nsamples, self.oldnpix, self.w, self.pixs
        )

        np.testing.assert_allclose(py_counts, cpp_counts, rtol=1.0e-5)
        np.testing.assert_allclose(py_mask, cpp_mask, rtol=1.0e-5)

    def test_initializeweights_pol2(self):
        (
            cpp_counts,
            cpp_sin,
            cpp_cos,
            cpp_sin2,
            cpp_cos2,
            cpp_sincos,
        ) = process_samples.py_process_pol2(
            self.nsamples, self.oldnpix, self.w, self.pixs, self.phi
        )

        (
            py_counts,
            py_sin,
            py_cos,
            py_sin2,
            py_cos2,
            py_sincos,
        ) = iw.initializeweights_pol2(
            self.nsamples, self.oldnpix, self.w, self.pixs, self.phi
        )

        # print(np.random.random(size=10))
        np.testing.assert_allclose(py_counts, cpp_counts, rtol=1.0e-5)
        np.testing.assert_allclose(py_sin, cpp_sin, rtol=1.0e-5)
        np.testing.assert_allclose(py_cos, cpp_cos, rtol=1.0e-5)
        np.testing.assert_allclose(py_sin2, cpp_sin2, rtol=1.0e-5)
        np.testing.assert_allclose(py_cos2, cpp_cos2, rtol=1.0e-5)
        np.testing.assert_allclose(py_sincos, cpp_sincos, rtol=1.0e-5)

    def test_initializeweights_pol3(self):
        (
            cpp_counts,
            cpp_sine,
            cpp_cosine,
            cpp_sin,
            cpp_cos,
            cpp_sin2,
            cpp_cos2,
            cpp_sincos,
        ) = process_samples.py_process_pol3(
            self.nsamples,
            self.oldnpix,
            self.w,
            self.pixs,
            self.phi,
        )

        (
            py_counts,
            py_sine,
            py_cosine,
            py_sin,
            py_cos,
            py_sin2,
            py_cos2,
            py_sincos,
        ) = iw.initializeweights_pol3(
            self.nsamples, self.oldnpix, self.w, self.pixs, self.phi
        )

        # print(np.random.random(size=10))
        np.testing.assert_allclose(py_counts, cpp_counts, rtol=1.0e-5)
        np.testing.assert_allclose(py_sine, cpp_sine, rtol=1.0e-5)
        np.testing.assert_allclose(py_cosine, cpp_cosine, rtol=1.0e-5)
        np.testing.assert_allclose(py_sin, cpp_sin, rtol=1.0e-5)
        np.testing.assert_allclose(py_cos, cpp_cos, rtol=1.0e-5)
        np.testing.assert_allclose(py_sin2, cpp_sin2, rtol=1.0e-5)
        np.testing.assert_allclose(py_cos2, cpp_cos2, rtol=1.0e-5)
        np.testing.assert_allclose(py_sincos, cpp_sincos, rtol=1.0e-5)


class TestInitializeWeights_float64:
    # Testing on random data: <https://numpy.org/doc/stable/reference/testing.html#tests-on-random-data>
    np.random.seed(414)
    oldnpix = 128
    nsamples = oldnpix * 10
    w = np.random.random(size=nsamples).astype(dtype=np.float64)
    pixs = np.random.randint(low=0, high=oldnpix, size=nsamples)
    badpixs = np.random.randint(low=0, high=nsamples, size=100)
    pixs[badpixs] = -1
    phi = np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=nsamples).astype(
        dtype=np.float64
    )

    def test_initializeweights_pol1(self):
        cpp_counts, cpp_mask = process_samples.py_process_pol1(
            self.nsamples, self.oldnpix, self.w, self.pixs
        )

        py_counts, py_mask = iw.initializeweights_pol1(
            self.nsamples, self.oldnpix, self.w, self.pixs
        )

        np.testing.assert_allclose(py_counts, cpp_counts)
        np.testing.assert_allclose(py_mask, cpp_mask)

    def test_initializeweights_pol2(self):
        (
            cpp_counts,
            cpp_sin,
            cpp_cos,
            cpp_sin2,
            cpp_cos2,
            cpp_sincos,
        ) = process_samples.py_process_pol2(
            self.nsamples, self.oldnpix, self.w, self.pixs, self.phi
        )

        (
            py_counts,
            py_sin,
            py_cos,
            py_sin2,
            py_cos2,
            py_sincos,
        ) = iw.initializeweights_pol2(
            self.nsamples, self.oldnpix, self.w, self.pixs, self.phi
        )

        np.testing.assert_allclose(py_counts, cpp_counts)
        np.testing.assert_allclose(py_sin, cpp_sin)
        np.testing.assert_allclose(py_cos, cpp_cos)
        np.testing.assert_allclose(py_sin2, cpp_sin2)
        np.testing.assert_allclose(py_cos2, cpp_cos2)
        np.testing.assert_allclose(py_sincos, cpp_sincos)

    def test_initializeweights_pol3(self):
        (
            cpp_counts,
            cpp_sine,
            cpp_cosine,
            cpp_sin,
            cpp_cos,
            cpp_sin2,
            cpp_cos2,
            cpp_sincos,
        ) = process_samples.py_process_pol3(
            self.nsamples,
            self.oldnpix,
            self.w,
            self.pixs,
            self.phi,
        )

        (
            py_counts,
            py_sine,
            py_cosine,
            py_sin,
            py_cos,
            py_sin2,
            py_cos2,
            py_sincos,
        ) = iw.initializeweights_pol3(
            self.nsamples, self.oldnpix, self.w, self.pixs, self.phi
        )

        np.testing.assert_allclose(py_counts, cpp_counts)
        np.testing.assert_allclose(py_sine, cpp_sine)
        np.testing.assert_allclose(py_cosine, cpp_cosine)
        np.testing.assert_allclose(py_sin, cpp_sin)
        np.testing.assert_allclose(py_cos, cpp_cos)
        np.testing.assert_allclose(py_sin2, cpp_sin2)
        np.testing.assert_allclose(py_cos2, cpp_cos2)
        np.testing.assert_allclose(py_sincos, cpp_sincos)


class TestGetMaskPol:
    np.random.seed(414)
    oldnpix = 256
    nsamples = oldnpix * 10
    w = np.random.random(size=nsamples)
    pixs = np.random.randint(low=0, high=oldnpix, size=nsamples)
    badpixs = np.random.randint(low=0, high=nsamples, size=100)
    pixs[badpixs] = -1
    phi = np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=nsamples)
    threshold = 1.0e3

    def test_get_mask_pol2(self):
        counts, __, __, sin2, cos2, sincos = iw.initializeweights_pol2(
            self.nsamples, self.oldnpix, self.w, self.pixs, self.phi
        )

        cpp_mask_pol2 = process_samples.py_get_mask_pol(
            2, counts, sin2, cos2, sincos, self.threshold
        )

        py_mask_pol2, __ = iw.get_mask_pol(counts, sin2, cos2, sincos, self.threshold)

        np.testing.assert_array_equal(cpp_mask_pol2, py_mask_pol2)

    def test_get_mask_pol3(self):
        counts, __, __, __, __, sin2, cos2, sincos = iw.initializeweights_pol3(
            self.nsamples, self.oldnpix, self.w, self.pixs, self.phi
        )

        cpp_mask_pol3 = process_samples.py_get_mask_pol(
            3, counts, sin2, cos2, sincos, self.threshold
        )

        __, py_mask_pol3 = iw.get_mask_pol(counts, sin2, cos2, sincos, self.threshold)

        np.testing.assert_array_equal(cpp_mask_pol3, py_mask_pol3)


if __name__ == "__main__":
    test_class_initweigths_float32 = TestInitializeWeights_float32()
    print(f"Testing {type(test_class_initweigths_float32).__name__} ...")
    test_class_initweigths_float32.test_initializeweights_pol1()
    test_class_initweigths_float32.test_initializeweights_pol2()
    test_class_initweigths_float32.test_initializeweights_pol3()
    print(f"Testing {type(test_class_initweigths_float32).__name__} ... PASSED")

    test_class_initweights_float64 = TestInitializeWeights_float64()
    print(f"Testing {type(test_class_initweights_float64).__name__} ...")
    test_class_initweights_float64.test_initializeweights_pol1()
    test_class_initweights_float64.test_initializeweights_pol2()
    test_class_initweights_float64.test_initializeweights_pol3()
    print(f"Testing {type(test_class_initweights_float64).__name__} ... PASSED")

    test_class_get_mask_pol = TestGetMaskPol()
    print(f"Testing {type(test_class_get_mask_pol).__name__} ...")
    test_class_get_mask_pol.test_get_mask_pol2()
    test_class_get_mask_pol.test_get_mask_pol3()
    print(f"Testing {type(test_class_get_mask_pol).__name__} ... PASSED")
