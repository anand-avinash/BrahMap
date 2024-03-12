#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>

namespace py = pybind11;

template <typename dtype_float>
std::tuple<int, int, py::array_t<int>, py::array_t<dtype_float>,
           py::array_t<int>>
repixelization_pol1(int oldnpix, py::array_t<int> mask,
                    py::array_t<dtype_float> counts, py::array_t<int> obspix) {

  auto mask_ptr = mask.template unchecked<1>();
  auto counts_ptr = counts.template mutable_unchecked<1>();
  auto obspix_ptr = obspix.template mutable_unchecked<1>();

  py::array_t<int> old2new(oldnpix);
  auto old2new_ptr = old2new.template mutable_unchecked<1>();

  int n_new_pix = 0, n_removed_pix = 0;
  for (ssize_t pix = 0; pix < oldnpix; ++pix) {
    bool boolval = 0;

    for (ssize_t idx = 0; idx < mask.size(); ++idx) {

      if (mask_ptr(idx) == pix) {
        boolval = 1;
        break;
      } else {
        continue;
      } // if
    }   // for

    if (boolval) {
      old2new_ptr(pix) = n_new_pix;
      counts_ptr(n_new_pix) = counts_ptr(pix);
      obspix_ptr(n_new_pix) = obspix_ptr(pix);
      n_new_pix += 1;
    } else {
      old2new_ptr(pix) = -1;
      n_removed_pix += 1;
    } // if boolval

  } // for

  return std::make_tuple(n_new_pix, n_removed_pix, old2new, counts, obspix);

} // repixelization_pol1()

template <typename dtype_float>
std::tuple<int, int, py::array_t<int>, py::array_t<dtype_float>,
           py::array_t<int>, py::array_t<dtype_float>, py::array_t<dtype_float>,
           py::array_t<dtype_float>>
repixelization_pol2(int oldnpix, py::array_t<int> mask,
                    py::array_t<dtype_float> counts, py::array_t<int> obspix,
                    py::array_t<dtype_float> sin2,
                    py::array_t<dtype_float> cos2,
                    py::array_t<dtype_float> sincos) {
  auto mask_ptr = mask.template unchecked<1>();
  auto counts_ptr = counts.template mutable_unchecked<1>();
  auto obspix_ptr = obspix.template mutable_unchecked<1>();
  auto sin2_ptr = sin2.template mutable_unchecked<1>();
  auto cos2_ptr = cos2.template mutable_unchecked<1>();
  auto sincos_ptr = sincos.template mutable_unchecked<1>();

  py::array_t<int> old2new(oldnpix);
  auto old2new_ptr = old2new.template mutable_unchecked<1>();

  int n_new_pix = 0, n_removed_pix = 0;
  for (ssize_t pix = 0; pix < oldnpix; ++pix) {
    bool boolval = 0;

    for (ssize_t idx = 0; idx < mask.size(); ++idx) {

      if (mask_ptr(idx) == pix) {
        boolval = 1;
        break;
      } else {
        continue;
      } // if
    }   // for

    if (boolval) {
      old2new_ptr(pix) = n_new_pix;
      counts_ptr(n_new_pix) = counts_ptr(pix);
      obspix_ptr(n_new_pix) = obspix_ptr(pix);
      sin2_ptr(n_new_pix) = sin2_ptr(pix);
      cos2_ptr(n_new_pix) = cos2_ptr(pix);
      sincos_ptr(n_new_pix) = sincos_ptr(pix);
      n_new_pix += 1;
    } else {
      old2new_ptr(pix) = -1;
      n_removed_pix += 1;
    } // if boolval

  } // for

  return std::make_tuple(n_new_pix, n_removed_pix, old2new, counts, obspix,
                         sin2, cos2, sincos);
} // repixelization_pol2()

template <typename dtype_float>
std::tuple<int, int, py::array_t<int>, py::array_t<dtype_float>,
           py::array_t<int>, py::array_t<dtype_float>, py::array_t<dtype_float>,
           py::array_t<dtype_float>, py::array_t<dtype_float>,
           py::array_t<dtype_float>>
repixelization_pol3(int oldnpix, py::array_t<int> mask,
                    py::array_t<dtype_float> counts, py::array_t<int> obspix,
                    py::array_t<dtype_float> sin2,
                    py::array_t<dtype_float> cos2,
                    py::array_t<dtype_float> sincos,
                    py::array_t<dtype_float> sine,
                    py::array_t<dtype_float> cosine) {
  auto mask_ptr = mask.template unchecked<1>();
  auto counts_ptr = counts.template mutable_unchecked<1>();
  auto obspix_ptr = obspix.template mutable_unchecked<1>();
  auto sin2_ptr = sin2.template mutable_unchecked<1>();
  auto cos2_ptr = cos2.template mutable_unchecked<1>();
  auto sincos_ptr = sincos.template mutable_unchecked<1>();
  auto sine_ptr = sine.template mutable_unchecked<1>();
  auto cosine_ptr = cosine.template mutable_unchecked<1>();

  py::array_t<int> old2new(oldnpix);
  auto old2new_ptr = old2new.template mutable_unchecked<1>();

  int n_new_pix = 0, n_removed_pix = 0;
  for (ssize_t pix = 0; pix < oldnpix; ++pix) {
    bool boolval = 0;

    for (ssize_t idx = 0; idx < mask.size(); ++idx) {

      if (mask_ptr(idx) == pix) {
        boolval = 1;
        break;
      } else {
        continue;
      } // if
    }   // for

    if (boolval) {
      old2new_ptr(pix) = n_new_pix;
      counts_ptr(n_new_pix) = counts_ptr(pix);
      obspix_ptr(n_new_pix) = obspix_ptr(pix);
      sin2_ptr(n_new_pix) = sin2_ptr(pix);
      cos2_ptr(n_new_pix) = cos2_ptr(pix);
      sincos_ptr(n_new_pix) = sincos_ptr(pix);
      sine_ptr(n_new_pix) = sine_ptr(pix);
      cosine_ptr(n_new_pix) = cosine_ptr(pix);
      n_new_pix += 1;
    } else {
      old2new_ptr(pix) = -1;
      n_removed_pix += 1;
    } // if boolval

  } // for

  return std::make_tuple(n_new_pix, n_removed_pix, old2new, counts, obspix,
                         sin2, cos2, sincos, sine, cosine);
} // repixelization_pol3()

PYBIND11_MODULE(repixelize, m) {
  m.doc() = "repixelize";
  m.def("py_repixelization_pol1", &repixelization_pol1<float>, "test");
  m.def("py_repixelization_pol1", &repixelization_pol1<double>, "test");
  m.def("py_repixelization_pol2", &repixelization_pol2<float>, "test");
  m.def("py_repixelization_pol2", &repixelization_pol2<double>, "test");
  m.def("py_repixelization_pol3", &repixelization_pol3<float>, "test");
  m.def("py_repixelization_pol3", &repixelization_pol3<double>, "test");
}
