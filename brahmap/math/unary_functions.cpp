#include <cmath>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#ifndef _DISABLE_OMP
#include <omp.h>
#endif

namespace nb = nanobind;

// Alias for the signature of a general unary function pointer.
// Here the unary function refers to a function that takes one
// input and returns one output.
template <typename dfloat> using dfunc = dfloat (*)(const dfloat);

// Templated function to call a general unary function `unary` over an array
template <typename dfloat, dfunc<dfloat> unary>
void execute(                     //
    const ssize_t size,           //
    const dfloat *__restrict vec, //
    dfloat *__restrict result     //
) {

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < size; ++idx) {
    result[idx] = unary(vec[idx]);
  } // for

  return;
} // execute()

template <typename dfloat, dfunc<dfloat> unary, typename device> //
void register_unary_function(nb::module_ &m, const char *name) {
  using arr_t = nb::ndarray<dfloat, nb::ndim<1>, device, nb::c_contig>;

  m.def(
      name,                   //
      [](                     //
          const ssize_t size, //
          const arr_t vec,    //
          arr_t result        //
      ) {
        execute<dfloat, unary>( //
            size,               //
            vec.data(),         //
            result.data()       //
        );
      },
      nb::arg("size"),              //
      nb::arg("vec").noconvert(),   //
      nb::arg("result").noconvert() //
  );
}

NB_MODULE(unary_functions, m) {
  m.doc() = "unary_functions";

  // sin function
  register_unary_function<double, std::sin, nb::device::cpu>(m, "sin");
  register_unary_function<float, std::sin, nb::device::cpu>(m, "sin");

  // cos function
  register_unary_function<double, std::cos, nb::device::cpu>(m, "cos");
  register_unary_function<float, std::cos, nb::device::cpu>(m, "cos");

  // tan function
  register_unary_function<double, std::tan, nb::device::cpu>(m, "tan");
  register_unary_function<float, std::tan, nb::device::cpu>(m, "tan");

  // arcsin function
  register_unary_function<double, std::asin, nb::device::cpu>(m, "arcsin");
  register_unary_function<float, std::asin, nb::device::cpu>(m, "arcsin");

  // arccos function
  register_unary_function<double, std::acos, nb::device::cpu>(m, "arccos");
  register_unary_function<float, std::acos, nb::device::cpu>(m, "arccos");

  // arctan function
  register_unary_function<double, std::atan, nb::device::cpu>(m, "arctan");
  register_unary_function<float, std::atan, nb::device::cpu>(m, "arctan");

  // exp function
  register_unary_function<double, std::exp, nb::device::cpu>(m, "exp");
  register_unary_function<float, std::exp, nb::device::cpu>(m, "exp");

  // exp2 function
  register_unary_function<double, std::exp2, nb::device::cpu>(m, "exp2");
  register_unary_function<float, std::exp2, nb::device::cpu>(m, "exp2");

  // log function
  register_unary_function<double, std::log, nb::device::cpu>(m, "log");
  register_unary_function<float, std::log, nb::device::cpu>(m, "log");

  // log2 function
  register_unary_function<double, std::log2, nb::device::cpu>(m, "log2");
  register_unary_function<float, std::log2, nb::device::cpu>(m, "log2");

  // sqrt function
  register_unary_function<double, std::sqrt, nb::device::cpu>(m, "sqrt");
  register_unary_function<float, std::sqrt, nb::device::cpu>(m, "sqrt");

  // cbrt function
  register_unary_function<double, std::cbrt, nb::device::cpu>(m, "cbrt");
  register_unary_function<float, std::cbrt, nb::device::cpu>(m, "cbrt");
}
