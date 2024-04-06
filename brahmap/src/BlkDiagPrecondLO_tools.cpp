#include <functional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

template <typename dfloat>
void BDPLO_mult_QU(                //
    const ssize_t new_npix,        //
    const dfloat *weighted_sin_sq, //
    const dfloat *weighted_cos_sq, //
    const dfloat *weighted_sincos, //
    const dfloat *vec,             //
    dfloat *prod                   //
) {

  for (ssize_t idx = 0; idx < new_npix; ++idx) {
    dfloat one_over_determinant =
        1.0 / (weighted_sin_sq[idx] * weighted_cos_sq[idx] -
               weighted_sincos[idx] * weighted_sincos[idx]);

    if (std::abs(one_over_determinant) < 1.e+5) {
      prod[2 * idx] =
          one_over_determinant * (weighted_sin_sq[idx] * vec[2 * idx] -
                                  weighted_sincos[idx] * vec[2 * idx + 1]);
      prod[2 * idx + 1] =
          one_over_determinant * (weighted_cos_sq[idx] * vec[2 * idx + 1] -
                                  weighted_sincos[idx] * vec[2 * idx]);
    } // if
  }   // for

  return;
} // BDPLO_mult_QU()

template <typename dfloat>
void BDPLO_mult_IQU(               //
    const ssize_t new_npix,        //
    const dfloat *weighted_counts, //
    const dfloat *weighted_sin_sq, //
    const dfloat *weighted_cos_sq, //
    const dfloat *weighted_sincos, //
    const dfloat *weighted_sin,    //
    const dfloat *weighted_cos,    //
    const dfloat *vec,             //
    dfloat *prod                   //
) {

  for (ssize_t idx = 0; idx < new_npix; ++idx) {
    dfloat one_over_determinant =
        weighted_counts[idx] * weighted_cos_sq[idx] * weighted_sin_sq[idx] +
        2.0 * weighted_cos[idx] * weighted_sin[idx] * weighted_sincos[idx] -
        weighted_counts[idx] * weighted_sincos[idx] * weighted_sincos[idx] -
        weighted_cos[idx] * weighted_cos[idx] * weighted_sin_sq[idx] -
        weighted_sin[idx] * weighted_sin[idx] * weighted_cos_sq[idx];

    one_over_determinant = 1.0 / one_over_determinant;

    if (std::abs(one_over_determinant) < 1.e+5) {
      prod[3 * idx] = one_over_determinant *
                      ((weighted_cos_sq[idx] * weighted_sin_sq[idx] -
                        weighted_sincos[idx] * weighted_sincos[idx]) *
                           vec[3 * idx] +
                       (weighted_sin[idx] * weighted_sincos[idx] -
                        weighted_cos[idx] * weighted_sin_sq[idx]) *
                           vec[3 * idx + 1] +
                       (weighted_cos[idx] * weighted_sincos[idx] -
                        weighted_sin[idx] * weighted_cos_sq[idx]) *
                           vec[3 * idx + 2]);
      prod[3 * idx + 1] = one_over_determinant *
                          ((weighted_sin[idx] * weighted_sincos[idx] -
                            weighted_cos[idx] * weighted_sin_sq[idx]) *
                               vec[3 * idx] +
                           (weighted_counts[idx] * weighted_sin_sq[idx] -
                            weighted_sin[idx] * weighted_sin[idx]) *
                               vec[3 * idx + 1] +
                           (weighted_sin[idx] * weighted_cos[idx] -
                            weighted_counts[idx] * weighted_sincos[idx]) *
                               vec[3 * idx + 2]);
      prod[3 * idx + 2] = one_over_determinant *
                          ((weighted_cos[idx] * weighted_sincos[idx] -
                            weighted_sin[idx] * weighted_cos_sq[idx]) *
                               vec[3 * idx] +
                           (weighted_cos[idx] * weighted_sin[idx] -
                            weighted_counts[idx] * weighted_sincos[idx]) *
                               vec[3 * idx + 1] +
                           (weighted_counts[idx] * weighted_cos_sq[idx] -
                            weighted_cos[idx] * weighted_cos[idx]) *
                               vec[3 * idx + 2]);
    } // if
  }   // for

  return;
} // BDPLO_mult_IQU()

template <typename dfloat>
std::function<void(                            //
    const ssize_t new_npix,                    //
    const py::array_t<dfloat> weighted_sin_sq, //
    const py::array_t<dfloat> weighted_cos_sq, //
    const py::array_t<dfloat> weighted_sincos, //
    const py::array_t<dfloat> vec,             //
    py::array_t<dfloat> prod                   //
    )>
    numpy_bind_BDPLO_mult_QU =           //
    [](const ssize_t new_npix,           //
       const py::buffer weighted_sin_sq, //
       const py::buffer weighted_cos_sq, //
       const py::buffer weighted_sincos, //
       const py::buffer vec,             //
       py::buffer prod                   //
    ) {
      py::buffer_info weighted_sin_sq_info = weighted_sin_sq.request();
      py::buffer_info weighted_cos_sq_info = weighted_cos_sq.request();
      py::buffer_info weighted_sincos_info = weighted_sincos.request();
      py::buffer_info vec_info = vec.request();
      py::buffer_info prod_info = prod.request();

      const dfloat *weighted_sin_sq_ptr =
          reinterpret_cast<dfloat *>(weighted_sin_sq_info.ptr);
      const dfloat *weighted_cos_sq_ptr =
          reinterpret_cast<dfloat *>(weighted_cos_sq_info.ptr);
      const dfloat *weighted_sincos_ptr =
          reinterpret_cast<dfloat *>(weighted_sincos_info.ptr);
      const dfloat *vec_ptr = reinterpret_cast<dfloat *>(vec_info.ptr);
      dfloat *prod_ptr = reinterpret_cast<dfloat *>(prod_info.ptr);

      BDPLO_mult_QU(           //
          new_npix,            //
          weighted_sin_sq_ptr, //
          weighted_cos_sq_ptr, //
          weighted_sincos_ptr, //
          vec_ptr,             //
          prod_ptr             //
      );

      return;
    }; // numpy_bind_BDPLO_mult_QU()

template <typename dfloat>
std::function<void(                            //
    const ssize_t new_npix,                    //
    const py::array_t<dfloat> weighted_counts, //
    const py::array_t<dfloat> weighted_sin_sq, //
    const py::array_t<dfloat> weighted_cos_sq, //
    const py::array_t<dfloat> weighted_sincos, //
    const py::array_t<dfloat> weighted_sin,    //
    const py::array_t<dfloat> weighted_cos,    //
    const py::array_t<dfloat> vec,             //
    py::array_t<dfloat> prod                   //
    )>
    numpy_bind_BDPLO_mult_IQU =          //
    [](const ssize_t new_npix,           //
       const py::buffer weighted_counts, //
       const py::buffer weighted_sin_sq, //
       const py::buffer weighted_cos_sq, //
       const py::buffer weighted_sincos, //
       const py::buffer weighted_sin,    //
       const py::buffer weighted_cos,    //
       const py::buffer vec,             //
       py::buffer prod                   //
    ) {
      py::buffer_info weighted_counts_info = weighted_counts.request();
      py::buffer_info weighted_sin_sq_info = weighted_sin_sq.request();
      py::buffer_info weighted_cos_sq_info = weighted_cos_sq.request();
      py::buffer_info weighted_sincos_info = weighted_sincos.request();
      py::buffer_info weighted_sin_info = weighted_sin.request();
      py::buffer_info weighted_cos_info = weighted_cos.request();
      py::buffer_info vec_info = vec.request();
      py::buffer_info prod_info = prod.request();

      const dfloat *weighted_counts_ptr =
          reinterpret_cast<const dfloat *>(weighted_counts_info.ptr);
      const dfloat *weighted_sin_sq_ptr =
          reinterpret_cast<const dfloat *>(weighted_sin_sq_info.ptr);
      const dfloat *weighted_cos_sq_ptr =
          reinterpret_cast<const dfloat *>(weighted_cos_sq_info.ptr);
      const dfloat *weighted_sincos_ptr =
          reinterpret_cast<const dfloat *>(weighted_sincos_info.ptr);
      const dfloat *weighted_sin_ptr =
          reinterpret_cast<const dfloat *>(weighted_sin_info.ptr);
      const dfloat *weighted_cos_ptr =
          reinterpret_cast<const dfloat *>(weighted_cos_info.ptr);
      const dfloat *vec_ptr = reinterpret_cast<const dfloat *>(vec_info.ptr);
      dfloat *prod_ptr = reinterpret_cast<dfloat *>(prod_info.ptr);

      BDPLO_mult_IQU(          //
          new_npix,            //
          weighted_counts_ptr, //
          weighted_sin_sq_ptr, //
          weighted_cos_sq_ptr, //
          weighted_sincos_ptr, //
          weighted_sin_ptr,    //
          weighted_cos_ptr,    //
          vec_ptr,             //
          prod_ptr             //
      );

      return;
    }; // numpy_bind_BDPLO_mult_IQU()

PYBIND11_MODULE(BlkDiagPrecondLO_tools, m) {
  m.doc() = "BlkDiagPrecondLO_tools";
  m.def("BDPLO_mult_QU", numpy_bind_BDPLO_mult_QU<float>, //
        py::arg("new_npix"),                              //
        py::arg("weighted_sin_sq").noconvert(),           //
        py::arg("weighted_cos_sq").noconvert(),           //
        py::arg("weighted_sincos").noconvert(),           //
        py::arg("vec").noconvert(),                       //
        py::arg("prod").noconvert()                       //
  );
  m.def("BDPLO_mult_QU", numpy_bind_BDPLO_mult_QU<double>, //
        py::arg("new_npix"),                               //
        py::arg("weighted_sin_sq").noconvert(),            //
        py::arg("weighted_cos_sq").noconvert(),            //
        py::arg("weighted_sincos").noconvert(),            //
        py::arg("vec").noconvert(),                        //
        py::arg("prod").noconvert()                        //
  );

  m.def("BDPLO_mult_IQU", numpy_bind_BDPLO_mult_IQU<float>, //
        py::arg("new_npix"),                                //
        py::arg("weighted_counts").noconvert(),             //
        py::arg("weighted_sin_sq").noconvert(),             //
        py::arg("weighted_cos_sq").noconvert(),             //
        py::arg("weighted_sincos").noconvert(),             //
        py::arg("weighted_sin").noconvert(),                //
        py::arg("weighted_cos").noconvert(),                //
        py::arg("vec").noconvert(),                         //
        py::arg("prod").noconvert()                         //
  );
  m.def("BDPLO_mult_IQU", numpy_bind_BDPLO_mult_IQU<double>, //
        py::arg("new_npix"),                                 //
        py::arg("weighted_counts").noconvert(),              //
        py::arg("weighted_sin_sq").noconvert(),              //
        py::arg("weighted_cos_sq").noconvert(),              //
        py::arg("weighted_sincos").noconvert(),              //
        py::arg("weighted_sin").noconvert(),                 //
        py::arg("weighted_cos").noconvert(),                 //
        py::arg("vec").noconvert(),                          //
        py::arg("prod").noconvert()                          //
  );
}
