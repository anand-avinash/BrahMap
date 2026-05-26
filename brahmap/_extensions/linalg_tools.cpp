#include <functional>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#ifndef _DISABLE_OMP
#include <omp.h>
#endif

namespace nb = nanobind;

template <typename dfloat>
void multiply_array(               //
    const ssize_t nsamples,        //
    const dfloat *__restrict diag, //
    const dfloat *__restrict vec,  //
    dfloat *__restrict prod        //
) {

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < nsamples; ++idx) {
    prod[idx] = diag[idx] * vec[idx];
  } // for
  return;
} // multiply_array()

template <typename dfloat, typename device> //
void register_multiply_array(nb::module_ &m) {
  using arr_t = nb::ndarray<dfloat, nb::ndim<1>, device, nb::c_contig>;

  m.def(
      "multiply_array",           //
      [](                         //
          const ssize_t nsamples, //
          const arr_t diag,       //
          const arr_t vec,        //
          arr_t prod              //
      ) {
        multiply_array(  //
            nsamples,    //
            diag.data(), //
            vec.data(),  //
            prod.data()  //
        );
      },
      nb::arg("nsamples"),         //
      nb::arg("diag").noconvert(), //
      nb::arg("vec").noconvert(),  //
      nb::arg("prod").noconvert()  //
  );
}

NB_MODULE(linalg_tools, m) {
  m.doc() = "linalg_tools";

  register_multiply_array<double, nb::device::cpu>(m);
  register_multiply_array<float, nb::device::cpu>(m);
}