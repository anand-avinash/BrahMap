#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>

namespace py = pybind11;

template <typename dtype_float>
py::array_t<dtype_float> BlkDiagPrecondLO_mult_qu(
    int npix, py::array_t<dtype_float> sin2, py::array_t<dtype_float> cos2,
    py::array_t<dtype_float> sincos, py::array_t<dtype_float> vec) {

  auto sin2_ptr = sin2.template unchecked<1>();
  auto cos2_ptr = cos2.template unchecked<1>();
  auto sincos_ptr = sincos.template unchecked<1>();
  auto vec_ptr = vec.template unchecked<1>();

  int vec_size = vec.size();
  int oldnpix = sin2.size();
  py::array_t<dtype_float> prod(vec_size);
  py::array_t<dtype_float> determinant(oldnpix);
  py::array_t<bool> mask(oldnpix);
  auto prod_ptr = prod.template mutable_unchecked<1>();
  auto determinant_ptr = determinant.template mutable_unchecked<1>();
  auto mask_ptr = mask.template mutable_unchecked<1>();

  for (ssize_t idx = 0; idx < vec_size; ++idx) {
    prod_ptr(idx) = 0.0;
  } // for

  for (ssize_t idx = 0; idx < oldnpix; ++idx) {
    mask_ptr(idx) = false;
    determinant_ptr(idx) =
        (cos2_ptr(idx) * sin2_ptr(idx)) - (sincos_ptr(idx) * sincos_ptr(idx));
    if (std::abs(determinant_ptr(idx)) > 1.e-5) {
      mask_ptr(idx) = true;
    } // if
  }   // for

  for (ssize_t idx = 0; idx < npix; ++idx) {
    if (mask_ptr(idx)) {
      prod_ptr(2 * idx) = (sin2_ptr(idx) * vec_ptr(2 * idx) -
                           sincos_ptr(idx) * vec_ptr(2 * idx + 1)) /
                          determinant_ptr(idx);
      prod_ptr(2 * idx + 1) = (cos2_ptr(idx) * vec_ptr(2 * idx + 1) -
                               sincos_ptr(idx) * vec_ptr(2 * idx)) /
                              determinant_ptr(idx);
    } else {
      continue;
    } // if
  }   // for

  return prod;

} // BlkDiagPrecondLO_mult_qu()

template <typename dtype_float>
py::array_t<dtype_float> BlkDiagPrecondLO_mult_iqu(
    int npix, py::array_t<dtype_float> counts, py::array_t<dtype_float> sine,
    py::array_t<dtype_float> cosine, py::array_t<dtype_float> sin2,
    py::array_t<dtype_float> cos2, py::array_t<dtype_float> sincos,
    py::array_t<dtype_float> vec) {

  auto counts_ptr = counts.template unchecked<1>();
  auto sine_ptr = sine.template unchecked<1>();
  auto cosine_ptr = cosine.template unchecked<1>();
  auto sin2_ptr = sin2.template unchecked<1>();
  auto cos2_ptr = cos2.template unchecked<1>();
  auto sincos_ptr = sincos.template unchecked<1>();
  auto vec_ptr = vec.template unchecked<1>();

  int vec_size = vec.size();
  int oldnpix = counts.size();
  py::array_t<dtype_float> prod(vec_size);
  py::array_t<dtype_float> determinant(oldnpix);
  py::array_t<bool> mask(oldnpix);
  auto prod_ptr = prod.template mutable_unchecked<1>();
  auto determinant_ptr = determinant.template mutable_unchecked<1>();
  auto mask_ptr = mask.template mutable_unchecked<1>();

  for (ssize_t idx = 0; idx < vec_size; ++idx) {
    prod_ptr(idx) = 0.0;
  } // for

  for (ssize_t idx = 0; idx < oldnpix; ++idx) {
    mask_ptr(idx) = false;
    determinant_ptr(idx) =
        counts_ptr(idx) * (cos2_ptr(idx) * sin2_ptr(idx) -
                           sincos_ptr(idx) * sincos_ptr(idx)) -
        cosine_ptr(idx) * cosine_ptr(idx) * sin2_ptr(idx) -
        sine_ptr(idx) * sine_ptr(idx) * cos2_ptr(idx) +
        2.0 * cosine_ptr(idx) * sine_ptr(idx) * sincos_ptr(idx);
    if (std::abs(determinant_ptr(idx)) > 1.e-5) {
      mask_ptr(idx) = true;
    } // if
  }   // for

  for (ssize_t idx = 0; idx < npix; ++idx) {
    if (mask_ptr(idx)) {
      prod_ptr(3 * idx) =
          ((cos2_ptr(idx) * sin2_ptr(idx) - sincos_ptr(idx) * sincos_ptr(idx)) *
               vec_ptr(3 * idx) +
           (sine_ptr(idx) * sincos_ptr(idx) - cosine_ptr(idx) * sin2_ptr(idx)) *
               vec_ptr(3 * idx + 1) +
           (cosine_ptr(idx) * sincos_ptr(idx) - sine_ptr(idx) * cos2_ptr(idx)) *
               vec_ptr(3 * idx + 2)) /
          determinant_ptr(idx);
      prod_ptr(3 * idx + 1) =
          ((sine_ptr(idx) * sincos_ptr(idx) - cosine_ptr(idx) * sin2_ptr(idx)) *
               vec_ptr(3 * idx) +
           (counts_ptr(idx) * sin2_ptr(idx) - sine_ptr(idx) * sine_ptr(idx)) *
               vec_ptr(3 * idx + 1) +
           (sine_ptr(idx) * cosine_ptr(idx) -
            counts_ptr(idx) * sincos_ptr(idx)) *
               vec_ptr(3 * idx + 2)) /
          determinant_ptr(idx);
      prod_ptr(3 * idx + 2) =
          ((cosine_ptr(idx) * sincos_ptr(idx) - sine_ptr(idx) * cos2_ptr(idx)) *
               vec_ptr(3 * idx) +
           (-counts_ptr(idx) * sincos_ptr(idx) +
            cosine_ptr(idx) * sine_ptr(idx)) *
               vec_ptr(3 * idx + 1) +
           (counts_ptr(idx) * cos2_ptr(idx) -
            cosine_ptr(idx) * cosine_ptr(idx)) *
               vec_ptr(3 * idx + 2)) /
          determinant_ptr(idx);
    } else {
      continue;
    } // if
  }   // for

  return prod;

} // BlkDiagPrecondLO_mult_iqu()

PYBIND11_MODULE(BlkDiagPrecondLO_tools, m) {
  m.doc() = "BlkDiagPrecondLO_tools";
  m.def("py_BlkDiagPrecondLO_mult_qu", &BlkDiagPrecondLO_mult_qu<float>,
        "test");
  m.def("py_BlkDiagPrecondLO_mult_qu", &BlkDiagPrecondLO_mult_qu<double>,
        "test");
  m.def("py_BlkDiagPrecondLO_mult_iqu", &BlkDiagPrecondLO_mult_iqu<float>,
        "test");
  m.def("py_BlkDiagPrecondLO_mult_iqu", &BlkDiagPrecondLO_mult_iqu<double>,
        "test");
}
