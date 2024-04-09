#include <functional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename dfloat>
void uncorrelated_mult(     //
    const ssize_t nsamples, //
    const dfloat *diag,     //
    const dfloat *vec,      //
    dfloat *prod            //
) {

  for (ssize_t idx = 0; idx < nsamples; ++idx) {
    prod[idx] = diag[idx] * vec[idx];
  } // for
  return;
} // uncorrelated_mult()

template <typename dfloat>
std::function<void(                 //
    const ssize_t nsamples,         //
    const py::array_t<dfloat> diag, //
    const py::array_t<dfloat> vec,  //
    py::array_t<dfloat> prod        //
    )>
    numpy_bind_uncorrelated_mult = //
    [](const ssize_t nsamples,     //
       const py::buffer diag,      //
       const py::buffer vec,       //
       py::buffer prod             //

    ) {
      py::buffer_info diag_info = diag.request();
      py::buffer_info vec_info = vec.request();
      py::buffer_info prod_info = prod.request();

      const dfloat *diag_ptr = reinterpret_cast<const dfloat *>(diag_info.ptr);
      const dfloat *vec_ptr = reinterpret_cast<const dfloat *>(vec_info.ptr);
      dfloat *prod_ptr = reinterpret_cast<dfloat *>(prod_info.ptr);

      uncorrelated_mult( //
          nsamples,      //
          diag_ptr,      //
          vec_ptr,       //
          prod_ptr       //
      );

      return;
    }; // numpy_bind_uncorrelated_mult()

PYBIND11_MODULE(InvNoiseCov_tools, m) {
  m.doc() = "InvNoiseCov_tools";
  m.def("uncorrelated_mult", numpy_bind_uncorrelated_mult<float>,
        py::arg("nsamples"),         //
        py::arg("diag").noconvert(), //
        py::arg("vec").noconvert(),  //
        py::arg("prod").noconvert()  //
  );
  m.def("uncorrelated_mult", numpy_bind_uncorrelated_mult<double>,
        py::arg("nsamples"),         //
        py::arg("diag").noconvert(), //
        py::arg("vec").noconvert(),  //
        py::arg("prod").noconvert()  //
  );
}