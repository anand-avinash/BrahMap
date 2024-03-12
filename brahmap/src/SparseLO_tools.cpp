#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>

namespace py = pybind11;

template <typename dtype_float>
py::array_t<dtype_float> SparseLO_mult(int nrows, py::array_t<int> pixs,
                                       py::array_t<dtype_float> vec) {

  auto pixs_ptr = pixs.template unchecked<1>();
  auto vec_ptr = vec.template unchecked<1>();

  py::array_t<dtype_float> prod(nrows);
  auto prod_ptr = prod.template mutable_unchecked<1>();

  for (ssize_t idx = 0; idx < nrows; ++idx) {
    prod_ptr(idx) = 0.0;
  } // for

  for (ssize_t idx = 0; idx < nrows; ++idx) {
    if (pixs_ptr(idx) == -1)
      continue;
    prod_ptr(idx) += vec_ptr(pixs_ptr(idx));
  } // for

  return prod;

} // SparseLO_mult()

template <typename dtype_float>
py::array_t<dtype_float> SparseLO_rmult(int nrows, int ncols,
                                        py::array_t<int> pixs,
                                        py::array_t<dtype_float> vec) {

  auto pixs_ptr = pixs.template unchecked<1>();
  auto vec_ptr = vec.template unchecked<1>();

  py::array_t<dtype_float> prod(ncols);
  auto prod_ptr = prod.template mutable_unchecked<1>();

  for (ssize_t idx = 0; idx < ncols; ++idx) {
    prod_ptr(idx) = 0.0;
  } // for

  for (ssize_t idx = 0; idx < nrows; ++idx) {
    if (pixs_ptr(idx) == -1)
      continue;
    prod_ptr(pixs_ptr(idx)) += vec_ptr(idx);
  } // for

  return prod;

} // SparseLO_rmult

template <typename dtype_float>
py::array_t<dtype_float>
SparseLO_mult_qu(int nrows, py::array_t<int> pixs, py::array_t<dtype_float> sin,
                 py::array_t<dtype_float> cos, py::array_t<dtype_float> vec) {

  auto pixs_ptr = pixs.template unchecked<1>();
  auto sin_ptr = sin.template unchecked<1>();
  auto cos_ptr = cos.template unchecked<1>();
  auto vec_ptr = vec.template unchecked<1>();

  py::array_t<dtype_float> prod(nrows);
  auto prod_ptr = prod.template mutable_unchecked<1>();

  for (ssize_t idx = 0; idx < nrows; ++idx) {
    prod_ptr(idx) = 0.0;
  } // for

  for (ssize_t idx = 0; idx < nrows; ++idx) {
    if (pixs_ptr(idx) == -1)
      continue;
    prod_ptr(idx) += vec_ptr(2 * pixs_ptr(idx)) * cos_ptr(idx) +
                     vec_ptr(2 * pixs_ptr(idx) + 1) * sin_ptr(idx);
  } // for

  return prod;

} // SparseLO_mult_qu

template <typename dtype_float>
py::array_t<dtype_float>
SparseLO_rmult_qu(int nrows, int ncols, py::array_t<int> pixs,
                  py::array_t<dtype_float> sin, py::array_t<dtype_float> cos,
                  py::array_t<dtype_float> vec) {

  auto pixs_ptr = pixs.template unchecked<1>();
  auto sin_ptr = sin.template unchecked<1>();
  auto cos_ptr = cos.template unchecked<1>();
  auto vec_ptr = vec.template unchecked<1>();

  py::array_t<dtype_float> prod(ncols * 2);
  auto prod_ptr = prod.template mutable_unchecked<1>();

  for (ssize_t idx = 0; idx < ncols * 2; ++idx) {
    prod_ptr(idx) = 0.0;
  } // for

  for (ssize_t idx = 0; idx < nrows; ++idx) {
    if (pixs_ptr(idx) == -1)
      continue;
    prod_ptr(2 * pixs_ptr(idx)) += vec_ptr(idx) * cos_ptr(idx);
    prod_ptr(2 * pixs_ptr(idx) + 1) += vec_ptr(idx) * sin_ptr(idx);
  } // for

  return prod;

} // SparseLO_rmult_qu

template <typename dtype_float>
py::array_t<dtype_float> SparseLO_mult_iqu(int nrows, py::array_t<int> pixs,
                                           py::array_t<dtype_float> sin,
                                           py::array_t<dtype_float> cos,
                                           py::array_t<dtype_float> vec) {
  auto pixs_ptr = pixs.template unchecked<1>();
  auto sin_ptr = sin.template unchecked<1>();
  auto cos_ptr = cos.template unchecked<1>();
  auto vec_ptr = vec.template unchecked<1>();

  py::array_t<dtype_float> prod(nrows);
  auto prod_ptr = prod.template mutable_unchecked<1>();

  for (ssize_t idx = 0; idx < nrows; ++idx) {
    prod_ptr(idx) = 0.0;
  } // for

  for (ssize_t idx = 0; idx < nrows; ++idx) {

    if (pixs_ptr(idx) == -1)
      continue;

    prod_ptr(idx) += vec_ptr(3 * pixs_ptr(idx)) +
                     vec_ptr(3 * pixs_ptr(idx) + 1) * cos_ptr(idx) +
                     vec_ptr(3 * pixs_ptr(idx) + 2) * sin_ptr(idx);
  } // for

  return prod;

} // SparseLO_mult_iqu

template <typename dtype_float>
py::array_t<dtype_float>
SparseLO_rmult_iqu(int nrows, int ncols, py::array_t<int> pixs,
                   py::array_t<dtype_float> sin, py::array_t<dtype_float> cos,
                   py::array_t<dtype_float> vec) {
  auto pixs_ptr = pixs.template unchecked<1>();
  auto sin_ptr = sin.template unchecked<1>();
  auto cos_ptr = cos.template unchecked<1>();
  auto vec_ptr = vec.template unchecked<1>();

  py::array_t<dtype_float> prod(ncols * 3);
  auto prod_ptr = prod.template mutable_unchecked<1>();

  for (ssize_t idx = 0; idx < ncols * 3; ++idx) {
    prod_ptr(idx) = 0.0;
  } // for

  for (ssize_t idx = 0; idx < nrows; ++idx) {

    if (pixs_ptr(idx) == -1)
      continue;

    prod_ptr(3 * pixs_ptr(idx)) += vec_ptr(idx);
    prod_ptr(3 * pixs_ptr(idx) + 1) += vec_ptr(idx) * cos_ptr(idx);
    prod_ptr(3 * pixs_ptr(idx) + 2) += vec_ptr(idx) * sin_ptr(idx);
  } // for

  return prod;
} // SparseLO_rmult_iqu

PYBIND11_MODULE(SparseLO_tools, m) {
  m.doc() = "SparseLO_tools";
  m.def("py_SparseLO_mult", &SparseLO_mult<float>, "test");
  m.def("py_SparseLO_mult", &SparseLO_mult<double>, "test");
  m.def("py_SparseLO_mult_qu", &SparseLO_mult_qu<float>, "test");
  m.def("py_SparseLO_mult_qu", &SparseLO_mult_qu<double>, "test");
  m.def("py_SparseLO_mult_iqu", &SparseLO_mult_iqu<float>, "test");
  m.def("py_SparseLO_mult_iqu", &SparseLO_mult_iqu<double>, "test");

  m.def("py_SparseLO_rmult", &SparseLO_rmult<float>, "test");
  m.def("py_SparseLO_rmult", &SparseLO_rmult<double>, "test");
  m.def("py_SparseLO_rmult_qu", &SparseLO_rmult_qu<float>, "test");
  m.def("py_SparseLO_rmult_qu", &SparseLO_rmult_qu<double>, "test");
  m.def("py_SparseLO_rmult_iqu", &SparseLO_rmult_iqu<float>, "test");
  m.def("py_SparseLO_rmult_iqu", &SparseLO_rmult_iqu<double>, "test");
}
