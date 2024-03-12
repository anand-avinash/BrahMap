#include <algorithm>
#include <cmath>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <tuple>
#include <vector>

namespace py = pybind11;

template <typename dtype_int, typename dtype_float>
std::tuple<py::array_t<dtype_float>, std::vector<int>>
process_pol1(int nsamples, int oldnpix, py::array_t<dtype_float> w,
             py::array_t<dtype_int> pixs) {

  auto w_ptr = w.template unchecked<1>();
  auto pixs_ptr = pixs.template unchecked<1>();

  dtype_int pixel;
  py::array_t<dtype_float> counts_arr(oldnpix);
  auto counts = counts_arr.template mutable_unchecked<1>();

  for (ssize_t idx = 0; idx < oldnpix; ++idx) {
    counts(idx) = 0.0;
  } // for

  for (ssize_t idx = 0; idx < nsamples; ++idx) {
    pixel = pixs_ptr(idx);
    if (pixel == -1)
      continue;
    counts(pixel) += w_ptr(idx);
  } // for

  std::vector<int> mask;
  for (ssize_t idx = 0; idx < oldnpix; ++idx) {
    if (counts(idx) > 0) {
      mask.push_back(idx);
    } // if
  }   // for

  return std::make_tuple(counts_arr, mask);

} // process_pol1()

template <typename dtype_int, typename dtype_float>
std::tuple<py::array_t<dtype_float>, py::array_t<dtype_float>,
           py::array_t<dtype_float>, py::array_t<dtype_float>,
           py::array_t<dtype_float>, py::array_t<dtype_float>>
process_pol2(int nsamples, int oldnpix, py::array_t<dtype_float> w,
             py::array_t<dtype_int> pixs, py::array_t<dtype_float> phi) {

  auto w_ptr = w.template unchecked<1>();
  auto pixs_ptr = pixs.template unchecked<1>();
  auto phi_ptr = phi.template unchecked<1>();

  py::array_t<dtype_float> counts_arr(oldnpix);
  py::array_t<dtype_float> sin_arr(phi.size());
  py::array_t<dtype_float> cos_arr(phi.size());
  py::array_t<dtype_float> sin2_arr(oldnpix);
  py::array_t<dtype_float> cos2_arr(oldnpix);
  py::array_t<dtype_float> sincos_arr(oldnpix);

  auto counts = counts_arr.template mutable_unchecked<1>();
  auto sin = sin_arr.template mutable_unchecked<1>();
  auto cos = cos_arr.template mutable_unchecked<1>();
  auto sin2 = sin2_arr.template mutable_unchecked<1>();
  auto cos2 = cos2_arr.template mutable_unchecked<1>();
  auto sincos = sincos_arr.template mutable_unchecked<1>();

  for (ssize_t i = 0; i < oldnpix; ++i) {
    counts(i) = 0.0;
    sin2(i) = 0.0;
    cos2(i) = 0.0;
    sincos(i) = 0.0;
  } // for

  for (ssize_t i = 0; i < phi.size(); ++i) {
    sin(i) = std::sin(2.0 * phi_ptr(i));
    cos(i) = std::cos(2.0 * phi_ptr(i));
  } // for

  for (ssize_t i = 0; i < nsamples; ++i) {
    int pixel = pixs_ptr(i);
    if (pixel == -1)
      continue;
    counts(pixel) += w_ptr(i);
    sin2(pixel) += w_ptr(i) * sin(i) * sin(i);
    cos2(pixel) += w_ptr(i) * cos(i) * cos(i);
    sincos(pixel) += w_ptr(i) * sin(i) * cos(i);
  } // for

  return std::make_tuple(counts_arr, sin_arr, cos_arr, sin2_arr, cos2_arr,
                         sincos_arr);

} // process_pol2()

template <typename dtype_int, typename dtype_float>
std::tuple<py::array_t<dtype_float>, py::array_t<dtype_float>,
           py::array_t<dtype_float>, py::array_t<dtype_float>,
           py::array_t<dtype_float>, py::array_t<dtype_float>,
           py::array_t<dtype_float>, py::array_t<dtype_float>>
process_pol3(int nsamples, int oldnpix, py::array_t<dtype_float> w,
             py::array_t<dtype_int> pixs, py::array_t<dtype_float> phi) {
  auto w_ptr = w.template unchecked<1>();
  auto pixs_ptr = pixs.template unchecked<1>();
  auto phi_ptr = phi.template unchecked<1>();

  py::array_t<dtype_float> counts_arr(oldnpix);
  py::array_t<dtype_float> sine_arr(oldnpix);
  py::array_t<dtype_float> cosine_arr(oldnpix);
  py::array_t<dtype_float> sin_arr(phi.size());
  py::array_t<dtype_float> cos_arr(phi.size());
  py::array_t<dtype_float> sin2_arr(oldnpix);
  py::array_t<dtype_float> cos2_arr(oldnpix);
  py::array_t<dtype_float> sincos_arr(oldnpix);

  auto counts = counts_arr.template mutable_unchecked<1>();
  auto sine = sine_arr.template mutable_unchecked<1>();
  auto cosine = cosine_arr.template mutable_unchecked<1>();
  auto sin = sin_arr.template mutable_unchecked<1>();
  auto cos = cos_arr.template mutable_unchecked<1>();
  auto sin2 = sin2_arr.template mutable_unchecked<1>();
  auto cos2 = cos2_arr.template mutable_unchecked<1>();
  auto sincos = sincos_arr.template mutable_unchecked<1>();

  for (ssize_t i = 0; i < oldnpix; ++i) {
    counts(i) = 0.0;
    sine(i) = 0.0;
    cosine(i) = 0.0;
    sin2(i) = 0.0;
    cos2(i) = 0.0;
    sincos(i) = 0.0;
  }

  for (ssize_t i = 0; i < phi.size(); ++i) {
    sin(i) = std::sin(2.0 * phi_ptr(i));
    cos(i) = std::cos(2.0 * phi_ptr(i));
  }

  for (ssize_t i = 0; i < nsamples; ++i) {
    int pixel = pixs_ptr(i);
    if (pixel == -1)
      continue;
    counts(pixel) += w_ptr(i);
    sine(pixel) += w_ptr(i) * sin(i);
    cosine(pixel) += w_ptr(i) * cos(i);
    sin2(pixel) += w_ptr(i) * sin(i) * sin(i);
    cos2(pixel) += w_ptr(i) * cos(i) * cos(i);
    sincos(pixel) += w_ptr(i) * sin(i) * cos(i);
  }

  return std::make_tuple(counts_arr, sine_arr, cosine_arr, sin_arr, cos_arr,
                         sin2_arr, cos2_arr, sincos_arr);
}

template <typename dtype_float>
std::vector<int>
get_mask_pol(int pol, py::array_t<dtype_float> counts,
             py::array_t<dtype_float> sin2, py::array_t<dtype_float> cos2,
             py::array_t<dtype_float> sincos, dtype_float threshold) {
  ssize_t oldnpix = counts.size();
  auto counts_ptr = counts.template unchecked<1>();
  auto sin2_ptr = sin2.template unchecked<1>();
  auto cos2_ptr = cos2.template unchecked<1>();
  auto sincos_ptr = sincos.template unchecked<1>();

  std::vector<int> mask;
  for (ssize_t idx = 0; idx < oldnpix; ++idx) {
    double det =
        cos2_ptr[idx] * sin2_ptr[idx] - sincos_ptr[idx] * sincos_ptr[idx];
    double trace = cos2_ptr[idx] + sin2_ptr[idx];
    double sqrtf = std::sqrt(trace * trace / 4.0 - det);
    double lambda_max = trace / 2.0 + sqrtf;
    double lambda_min = trace / 2.0 - sqrtf;
    double cond_num = abs(lambda_max / lambda_min);

    if (cond_num <= threshold) {
      mask.push_back(idx);
    }
  } // for

  if (pol == 2) {
    return mask;
  } // if pol==2

  if (pol == 3) {
    std::vector<int> count_mask;
    std::vector<int> final_mask;
    for (ssize_t idx = 0; idx < oldnpix; ++idx) {
      if (counts_ptr[idx] > 2) {
        count_mask.push_back(idx);
      } // if
    }   // for

    /*
    1d-intersection of `mask` and `count_mask`. Taken from
    <https://en.cppreference.com/w/cpp/algorithm/set_intersection>
    */
    std::sort(mask.begin(), mask.end());
    std::sort(count_mask.begin(), count_mask.end());
    std::set_intersection(mask.begin(), mask.end(), count_mask.begin(),
                          count_mask.end(), std::back_inserter(final_mask));
    return final_mask;
  } // if pol==3

  throw std::runtime_error("Invalid value of `pol` encountered");
} // get_mask_pol()

PYBIND11_MODULE(process_samples, m) {
  m.doc() = "process_samples";
  m.def("py_process_pol1", &process_pol1<int32_t, float>, "test");
  m.def("py_process_pol1", &process_pol1<int32_t, double>, "test");
  m.def("py_process_pol1", &process_pol1<int64_t, float>, "test");
  m.def("py_process_pol1", &process_pol1<int64_t, double>, "test");
  m.def("py_process_pol2", &process_pol2<int32_t, float>, "test");
  m.def("py_process_pol2", &process_pol2<int32_t, double>, "test");
  m.def("py_process_pol2", &process_pol2<int64_t, float>, "test");
  m.def("py_process_pol2", &process_pol2<int64_t, double>, "test");
  // m.def("py_process_pol3", &process_pol3, "test");
  m.def("py_process_pol3", &process_pol3<int32_t, float>, "test");
  m.def("py_process_pol3", &process_pol3<int32_t, double>, "test");
  m.def("py_process_pol3", &process_pol3<int64_t, float>, "test");
  m.def("py_process_pol3", &process_pol3<int64_t, double>, "test");
  m.def("py_get_mask_pol", &get_mask_pol<float>, "test");
  m.def("py_get_mask_pol", &get_mask_pol<double>, "test");
}
