#include <functional>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#ifndef _DISABLE_OMP
#include <omp.h>
#endif

namespace py = pybind11;

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

template <template <typename, int = py::array::c_style> class buffer_t,
          typename dfloat>
std::function<void(              //
    const ssize_t nsamples,      //
    const buffer_t<dfloat> diag, //
    const buffer_t<dfloat> vec,  //
    buffer_t<dfloat> prod        //
    )>
    numpy_bind_multiply_array = //
    [](const ssize_t nsamples,  //
       const py::buffer diag,   //
       const py::buffer vec,    //
       py::buffer prod          //

    ) {
      py::buffer_info diag_info = diag.request();
      py::buffer_info vec_info = vec.request();
      py::buffer_info prod_info = prod.request();

      const dfloat *diag_ptr = reinterpret_cast<const dfloat *>(diag_info.ptr);
      const dfloat *vec_ptr = reinterpret_cast<const dfloat *>(vec_info.ptr);
      dfloat *prod_ptr = reinterpret_cast<dfloat *>(prod_info.ptr);

      multiply_array( //
          nsamples,   //
          diag_ptr,   //
          vec_ptr,    //
          prod_ptr    //
      );

      return;
    }; // numpy_bind_multiply_array()

PYBIND11_MODULE(linalg_tools, m) {
  m.doc() = "linalg_tools";
  m.def("multiply_array", numpy_bind_multiply_array<py::array_t, float>,
        py::arg("nsamples"), //
        py::arg("diag"),     //
        py::arg("vec"),      //
        py::arg("prod")      //
  );
  m.def("multiply_array", numpy_bind_multiply_array<py::array_t, double>,
        py::arg("nsamples"), //
        py::arg("diag"),     //
        py::arg("vec"),      //
        py::arg("prod")      //
  );
}