#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename dint, typename dfloat>
void repixelize_pol_I(const ssize_t new_npix, //
                      const dint *pixel_mask, //
                      dfloat *weighted_counts //
) {

  for (ssize_t idx = 0; idx < new_npix; ++idx) {
    dint pixel = pixel_mask[idx];
    weighted_counts[idx] = weighted_counts[pixel];
  } // for

} // repixelize_pol_I()

template <typename dint, typename dfloat>
void repixelize_pol_QU(const ssize_t new_npix,  //
                       const dint *pixel_mask,  //
                       dfloat *weighted_counts, //
                       dfloat *weighted_sin_sq, //
                       dfloat *weighted_cos_sq, //
                       dfloat *weighted_sincos  //
) {

  for (ssize_t idx = 0; idx < new_npix; ++idx) {
    dint pixel = pixel_mask[idx];
    weighted_counts[idx] = weighted_counts[pixel];
    weighted_sin_sq[idx] = weighted_sin_sq[pixel];
    weighted_cos_sq[idx] = weighted_cos_sq[pixel];
    weighted_sincos[idx] = weighted_sincos[pixel];
  } // for

} // repixelize_pol_QU()

template <typename dint, typename dfloat>
void repixelize_pol_IQU(const ssize_t new_npix,  //
                        const dint *pixel_mask,  //
                        dfloat *weighted_counts, //
                        dfloat *weighted_sin_sq, //
                        dfloat *weighted_cos_sq, //
                        dfloat *weighted_sincos, //
                        dfloat *weighted_sin,    //
                        dfloat *weighted_cos     //
) {

  for (ssize_t idx = 0; idx < new_npix; ++idx) {
    dint pixel = pixel_mask[idx];
    weighted_counts[idx] = weighted_counts[pixel];
    weighted_sin_sq[idx] = weighted_sin_sq[pixel];
    weighted_cos_sq[idx] = weighted_cos_sq[pixel];
    weighted_sincos[idx] = weighted_sincos[pixel];
    weighted_sin[idx] = weighted_sin[pixel];
    weighted_cos[idx] = weighted_cos[pixel];
  } // for

} // repixelize_pol_IQU()

template <typename dint, typename dfloat>
std::function<void(                     //
    const ssize_t new_npix,             //
    const py::array_t<dint> pixel_mask, //
    py::array_t<dfloat> weighted_counts //
    )>
    numpy_bind_repixelize_pol_I =   //
    [](const ssize_t new_npix,      //
       const py::buffer pixel_mask, //
       py::buffer weighted_counts   //
    ) {
      py::buffer_info pixel_mask_info = pixel_mask.request();
      py::buffer_info weighted_counts_info = weighted_counts.request();

      const dint *pixel_mask_ptr =
          reinterpret_cast<const dint *>(pixel_mask_info.ptr);
      dfloat *weighted_counts_ptr =
          reinterpret_cast<dfloat *>(weighted_counts_info.ptr);

      repixelize_pol_I(       //
          new_npix,           //
          pixel_mask_ptr,     //
          weighted_counts_ptr //
      );
    }; // numpy_bind_repixelize_pol_I()

template <typename dint, typename dfloat>
std::function<void(                      //
    const ssize_t new_npix,              //
    const py::array_t<dint> pixel_mask,  //
    py::array_t<dfloat> weighted_counts, //
    py::array_t<dfloat> weighted_sin_sq, //
    py::array_t<dfloat> weighted_cos_sq, //
    py::array_t<dfloat> weighted_sincos  //
    )>
    numpy_bind_repixelize_pol_QU =  //
    [](const ssize_t new_npix,      //
       const py::buffer pixel_mask, //
       py::buffer weighted_counts,  //
       py::buffer weighted_sin_sq,  //
       py::buffer weighted_cos_sq,  //
       py::buffer weighted_sincos   //
    ) {
      py::buffer_info pixel_mask_info = pixel_mask.request();
      py::buffer_info weighted_counts_info = weighted_counts.request();
      py::buffer_info weighted_sin_sq_info = weighted_sin_sq.request();
      py::buffer_info weighted_cos_sq_info = weighted_cos_sq.request();
      py::buffer_info weighted_sincos_info = weighted_sincos.request();

      const dint *pixel_mask_ptr =
          reinterpret_cast<const dint *>(pixel_mask_info.ptr);
      dfloat *weighted_counts_ptr =
          reinterpret_cast<dfloat *>(weighted_counts_info.ptr);
      dfloat *weighted_sin_sq_ptr =
          reinterpret_cast<dfloat *>(weighted_sin_sq_info.ptr);
      dfloat *weighted_cos_sq_ptr =
          reinterpret_cast<dfloat *>(weighted_cos_sq_info.ptr);
      dfloat *weighted_sincos_ptr =
          reinterpret_cast<dfloat *>(weighted_sincos_info.ptr);

      repixelize_pol_QU(       //
          new_npix,            //
          pixel_mask_ptr,      //
          weighted_counts_ptr, //
          weighted_sin_sq_ptr, //
          weighted_cos_sq_ptr, //
          weighted_sincos_ptr  //
      );
    }; // numpy_bind_repixelize_pol_QU()

template <typename dint, typename dfloat>
std::function<void(                      //
    const ssize_t new_npix,              //
    const py::array_t<dint> pixel_mask,  //
    py::array_t<dfloat> weighted_counts, //
    py::array_t<dfloat> weighted_sin_sq, //
    py::array_t<dfloat> weighted_cos_sq, //
    py::array_t<dfloat> weighted_sincos, //
    py::array_t<dfloat> weighted_sin,    //
    py::array_t<dfloat> weighted_cos     //
    )>
    numpy_bind_repixelize_pol_IQU = //
    [](const ssize_t new_npix,      //
       const py::buffer pixel_mask, //
       py::buffer weighted_counts,  //
       py::buffer weighted_sin_sq,  //
       py::buffer weighted_cos_sq,  //
       py::buffer weighted_sincos,  //
       py::buffer weighted_sin,     //
       py::buffer weighted_cos      //
    ) {
      py::buffer_info pixel_mask_info = pixel_mask.request();
      py::buffer_info weighted_counts_info = weighted_counts.request();
      py::buffer_info weighted_sin_sq_info = weighted_sin_sq.request();
      py::buffer_info weighted_cos_sq_info = weighted_cos_sq.request();
      py::buffer_info weighted_sincos_info = weighted_sincos.request();
      py::buffer_info weighted_sin_info = weighted_sin.request();
      py::buffer_info weighted_cos_info = weighted_cos.request();

      const dint *pixel_mask_ptr =
          reinterpret_cast<const dint *>(pixel_mask_info.ptr);
      dfloat *weighted_counts_ptr =
          reinterpret_cast<dfloat *>(weighted_counts_info.ptr);
      dfloat *weighted_sin_sq_ptr =
          reinterpret_cast<dfloat *>(weighted_sin_sq_info.ptr);
      dfloat *weighted_cos_sq_ptr =
          reinterpret_cast<dfloat *>(weighted_cos_sq_info.ptr);
      dfloat *weighted_sincos_ptr =
          reinterpret_cast<dfloat *>(weighted_sincos_info.ptr);
      dfloat *weighted_sin_ptr =
          reinterpret_cast<dfloat *>(weighted_sin_info.ptr);
      dfloat *weighted_cos_ptr =
          reinterpret_cast<dfloat *>(weighted_cos_info.ptr);

      repixelize_pol_IQU(      //
          new_npix,            //
          pixel_mask_ptr,      //
          weighted_counts_ptr, //
          weighted_sin_sq_ptr, //
          weighted_cos_sq_ptr, //
          weighted_sincos_ptr, //
          weighted_sin_ptr,    //
          weighted_cos_ptr     //
      );
    }; // numpy_bind_repixelize_pol_IQU()

PYBIND11_MODULE(repixelize, m) {
  m.doc() = "repixelize";
  m.def("repixelize_pol_I", numpy_bind_repixelize_pol_I<int32_t, float>,
        py::arg("new_npix"),                   //
        py::arg("pixel_mask").noconvert(),     //
        py::arg("weighted_counts").noconvert() //
  );
  m.def("repixelize_pol_I", numpy_bind_repixelize_pol_I<int64_t, float>,
        py::arg("new_npix"),                   //
        py::arg("pixel_mask").noconvert(),     //
        py::arg("weighted_counts").noconvert() //
  );
  m.def("repixelize_pol_I", numpy_bind_repixelize_pol_I<int32_t, double>,
        py::arg("new_npix"),                   //
        py::arg("pixel_mask").noconvert(),     //
        py::arg("weighted_counts").noconvert() //
  );
  m.def("repixelize_pol_I", numpy_bind_repixelize_pol_I<int64_t, double>,
        py::arg("new_npix"),                   //
        py::arg("pixel_mask").noconvert(),     //
        py::arg("weighted_counts").noconvert() //
  );

  m.def("repixelize_pol_QU", numpy_bind_repixelize_pol_QU<int32_t, float>,
        py::arg("new_npix"),                    //
        py::arg("pixel_mask").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert()  //
  );
  m.def("repixelize_pol_QU", numpy_bind_repixelize_pol_QU<int64_t, float>,
        py::arg("new_npix"),                    //
        py::arg("pixel_mask").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert()  //
  );
  m.def("repixelize_pol_QU", numpy_bind_repixelize_pol_QU<int32_t, double>,
        py::arg("new_npix"),                    //
        py::arg("pixel_mask").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert()  //
  );
  m.def("repixelize_pol_QU", numpy_bind_repixelize_pol_QU<int64_t, double>,
        py::arg("new_npix"),                    //
        py::arg("pixel_mask").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert()  //
  );

  m.def("repixelize_pol_IQU", numpy_bind_repixelize_pol_IQU<int32_t, float>,
        py::arg("new_npix"),                    //
        py::arg("pixel_mask").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert(), //
        py::arg("weighted_sin").noconvert(),    //
        py::arg("weighted_cos").noconvert()     //
  );
  m.def("repixelize_pol_IQU", numpy_bind_repixelize_pol_IQU<int64_t, float>,
        py::arg("new_npix"),                    //
        py::arg("pixel_mask").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert(), //
        py::arg("weighted_sin").noconvert(),    //
        py::arg("weighted_cos").noconvert()     //
  );
  m.def("repixelize_pol_IQU", numpy_bind_repixelize_pol_IQU<int32_t, double>,
        py::arg("new_npix"),                    //
        py::arg("pixel_mask").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert(), //
        py::arg("weighted_sin").noconvert(),    //
        py::arg("weighted_cos").noconvert()     //
  );
  m.def("repixelize_pol_IQU", numpy_bind_repixelize_pol_IQU<int64_t, double>,
        py::arg("new_npix"),                    //
        py::arg("pixel_mask").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert(), //
        py::arg("weighted_sin").noconvert(),    //
        py::arg("weighted_cos").noconvert()     //
  );
}
