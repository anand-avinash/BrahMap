#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename dint, typename dfloat>
void repixelize_pol_I(      //
    const ssize_t new_npix, //
    const dint *pixel_mask, //
    dfloat *weighted_counts //
) {

  for (ssize_t idx = 0; idx < new_npix; ++idx) {
    dint pixel = pixel_mask[idx];
    weighted_counts[idx] = weighted_counts[pixel];
  } // for

  return;

} // repixelize_pol_I()

template <typename dint, typename dfloat>
void repixelize_pol_QU(          //
    const ssize_t new_npix,      //
    const dint *pixel_mask,      //
    dfloat *weighted_counts,     //
    dfloat *weighted_sin_sq,     //
    dfloat *weighted_cos_sq,     //
    dfloat *weighted_sincos,     //
    dfloat *one_over_determinant //
) {

  for (ssize_t idx = 0; idx < new_npix; ++idx) {
    dint pixel = pixel_mask[idx];
    weighted_counts[idx] = weighted_counts[pixel];
    weighted_sin_sq[idx] = weighted_sin_sq[pixel];
    weighted_cos_sq[idx] = weighted_cos_sq[pixel];
    weighted_sincos[idx] = weighted_sincos[pixel];
    one_over_determinant[idx] = 1.0 / one_over_determinant[pixel];
  } // for

  return;

} // repixelize_pol_QU()

template <typename dint, typename dfloat>
void repixelize_pol_IQU(         //
    const ssize_t new_npix,      //
    const dint *pixel_mask,      //
    dfloat *weighted_counts,     //
    dfloat *weighted_sin_sq,     //
    dfloat *weighted_cos_sq,     //
    dfloat *weighted_sincos,     //
    dfloat *weighted_sin,        //
    dfloat *weighted_cos,        //
    dfloat *one_over_determinant //
) {

  for (ssize_t idx = 0; idx < new_npix; ++idx) {
    dint pixel = pixel_mask[idx];
    weighted_counts[idx] = weighted_counts[pixel];
    weighted_sin_sq[idx] = weighted_sin_sq[pixel];
    weighted_cos_sq[idx] = weighted_cos_sq[pixel];
    weighted_sincos[idx] = weighted_sincos[pixel];
    weighted_sin[idx] = weighted_sin[pixel];
    weighted_cos[idx] = weighted_cos[pixel];
    one_over_determinant[idx] = 1.0 / one_over_determinant[pixel];
  } // for

  return;

} // repixelize_pol_IQU()

template <typename dint>
void flag_bad_pixel_samples(const ssize_t nsamples,    //
                            const bool *pixel_flag,    //
                            const dint *old2new_pixel, //
                            dint *pointings,           //
                            bool *pointings_flag       //
) {

  for (ssize_t idx = 0; idx < nsamples; ++idx) {
    dint pixel = pointings[idx];
    bool pixflag = pixel_flag[pixel];
    pointings[idx] = pixflag * old2new_pixel[pixel];
    pointings_flag[idx] &= pixflag;
  } // for

  return;
} //  flag_bad_pixel_samples()

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
std::function<void(                          //
    const ssize_t new_npix,                  //
    const py::array_t<dint> pixel_mask,      //
    py::array_t<dfloat> weighted_counts,     //
    py::array_t<dfloat> weighted_sin_sq,     //
    py::array_t<dfloat> weighted_cos_sq,     //
    py::array_t<dfloat> weighted_sincos,     //
    py::array_t<dfloat> one_over_determinant //
    )>
    numpy_bind_repixelize_pol_QU =     //
    [](const ssize_t new_npix,         //
       const py::buffer pixel_mask,    //
       py::buffer weighted_counts,     //
       py::buffer weighted_sin_sq,     //
       py::buffer weighted_cos_sq,     //
       py::buffer weighted_sincos,     //
       py::buffer one_over_determinant //
    ) {
      py::buffer_info pixel_mask_info = pixel_mask.request();
      py::buffer_info weighted_counts_info = weighted_counts.request();
      py::buffer_info weighted_sin_sq_info = weighted_sin_sq.request();
      py::buffer_info weighted_cos_sq_info = weighted_cos_sq.request();
      py::buffer_info weighted_sincos_info = weighted_sincos.request();
      py::buffer_info one_over_determinant_info =
          one_over_determinant.request();

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
      dfloat *one_over_determinant_ptr =
          reinterpret_cast<dfloat *>(one_over_determinant_info.ptr);

      repixelize_pol_QU(           //
          new_npix,                //
          pixel_mask_ptr,          //
          weighted_counts_ptr,     //
          weighted_sin_sq_ptr,     //
          weighted_cos_sq_ptr,     //
          weighted_sincos_ptr,     //
          one_over_determinant_ptr //
      );
    }; // numpy_bind_repixelize_pol_QU()

template <typename dint, typename dfloat>
std::function<void(                          //
    const ssize_t new_npix,                  //
    const py::array_t<dint> pixel_mask,      //
    py::array_t<dfloat> weighted_counts,     //
    py::array_t<dfloat> weighted_sin_sq,     //
    py::array_t<dfloat> weighted_cos_sq,     //
    py::array_t<dfloat> weighted_sincos,     //
    py::array_t<dfloat> weighted_sin,        //
    py::array_t<dfloat> weighted_cos,        //
    py::array_t<dfloat> one_over_determinant //
    )>
    numpy_bind_repixelize_pol_IQU =    //
    [](const ssize_t new_npix,         //
       const py::buffer pixel_mask,    //
       py::buffer weighted_counts,     //
       py::buffer weighted_sin_sq,     //
       py::buffer weighted_cos_sq,     //
       py::buffer weighted_sincos,     //
       py::buffer weighted_sin,        //
       py::buffer weighted_cos,        //
       py::buffer one_over_determinant //
    ) {
      py::buffer_info pixel_mask_info = pixel_mask.request();
      py::buffer_info weighted_counts_info = weighted_counts.request();
      py::buffer_info weighted_sin_sq_info = weighted_sin_sq.request();
      py::buffer_info weighted_cos_sq_info = weighted_cos_sq.request();
      py::buffer_info weighted_sincos_info = weighted_sincos.request();
      py::buffer_info weighted_sin_info = weighted_sin.request();
      py::buffer_info weighted_cos_info = weighted_cos.request();
      py::buffer_info one_over_determinant_info =
          one_over_determinant.request();

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
      dfloat *one_over_determinant_ptr =
          reinterpret_cast<dfloat *>(one_over_determinant_info.ptr);

      repixelize_pol_IQU(          //
          new_npix,                //
          pixel_mask_ptr,          //
          weighted_counts_ptr,     //
          weighted_sin_sq_ptr,     //
          weighted_cos_sq_ptr,     //
          weighted_sincos_ptr,     //
          weighted_sin_ptr,        //
          weighted_cos_ptr,        //
          one_over_determinant_ptr //
      );
    }; // numpy_bind_repixelize_pol_IQU()

template <typename dint>
std::function<void(                        //
    const ssize_t nsamples,                //
    const py::array_t<bool> pixel_flag,    //
    const py::array_t<dint> old2new_pixel, //
    py::array_t<dint> pointings,           //
    py::array_t<bool> poitnings_flag       //
    )>
    numpy_bind_flag_bad_pixel_samples = //
    [](const ssize_t nsamples,          //
       const py::buffer pixel_flag,     //
       const py::buffer old2new_pixel,  //
       py::buffer pointings,            //
       py::buffer pointings_flag        //
    ) {
      py::buffer_info pixel_flag_info = pixel_flag.request();
      py::buffer_info old2new_pixel_info = old2new_pixel.request();
      py::buffer_info pointings_info = pointings.request();
      py::buffer_info pointings_flag_info = pointings_flag.request();

      const bool *pixel_flag_ptr =
          reinterpret_cast<bool *>(pixel_flag_info.ptr);
      const dint *old2new_pixel_ptr =
          reinterpret_cast<const dint *>(old2new_pixel_info.ptr);
      dint *pointings_ptr = reinterpret_cast<dint *>(pointings_info.ptr);
      bool *pointings_flag_ptr =
          reinterpret_cast<bool *>(pointings_flag_info.ptr);

      flag_bad_pixel_samples(nsamples, pixel_flag_ptr, old2new_pixel_ptr,
                             pointings_ptr, pointings_flag_ptr);

      return;
    }; // numpy_bind_flag_bad_pixel_samples()

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
        py::arg("new_npix"),                        //
        py::arg("pixel_mask").noconvert(),          //
        py::arg("weighted_counts").noconvert(),     //
        py::arg("weighted_sin_sq").noconvert(),     //
        py::arg("weighted_cos_sq").noconvert(),     //
        py::arg("weighted_sincos").noconvert(),     //
        py::arg("one_over_determinant").noconvert() //
  );
  m.def("repixelize_pol_QU", numpy_bind_repixelize_pol_QU<int64_t, float>,
        py::arg("new_npix"),                        //
        py::arg("pixel_mask").noconvert(),          //
        py::arg("weighted_counts").noconvert(),     //
        py::arg("weighted_sin_sq").noconvert(),     //
        py::arg("weighted_cos_sq").noconvert(),     //
        py::arg("weighted_sincos").noconvert(),     //
        py::arg("one_over_determinant").noconvert() //
  );
  m.def("repixelize_pol_QU", numpy_bind_repixelize_pol_QU<int32_t, double>,
        py::arg("new_npix"),                        //
        py::arg("pixel_mask").noconvert(),          //
        py::arg("weighted_counts").noconvert(),     //
        py::arg("weighted_sin_sq").noconvert(),     //
        py::arg("weighted_cos_sq").noconvert(),     //
        py::arg("weighted_sincos").noconvert(),     //
        py::arg("one_over_determinant").noconvert() //
  );
  m.def("repixelize_pol_QU", numpy_bind_repixelize_pol_QU<int64_t, double>,
        py::arg("new_npix"),                        //
        py::arg("pixel_mask").noconvert(),          //
        py::arg("weighted_counts").noconvert(),     //
        py::arg("weighted_sin_sq").noconvert(),     //
        py::arg("weighted_cos_sq").noconvert(),     //
        py::arg("weighted_sincos").noconvert(),     //
        py::arg("one_over_determinant").noconvert() //
  );

  m.def("repixelize_pol_IQU", numpy_bind_repixelize_pol_IQU<int32_t, float>,
        py::arg("new_npix"),                        //
        py::arg("pixel_mask").noconvert(),          //
        py::arg("weighted_counts").noconvert(),     //
        py::arg("weighted_sin_sq").noconvert(),     //
        py::arg("weighted_cos_sq").noconvert(),     //
        py::arg("weighted_sincos").noconvert(),     //
        py::arg("weighted_sin").noconvert(),        //
        py::arg("weighted_cos").noconvert(),        //
        py::arg("one_over_determinant").noconvert() //
  );
  m.def("repixelize_pol_IQU", numpy_bind_repixelize_pol_IQU<int64_t, float>,
        py::arg("new_npix"),                        //
        py::arg("pixel_mask").noconvert(),          //
        py::arg("weighted_counts").noconvert(),     //
        py::arg("weighted_sin_sq").noconvert(),     //
        py::arg("weighted_cos_sq").noconvert(),     //
        py::arg("weighted_sincos").noconvert(),     //
        py::arg("weighted_sin").noconvert(),        //
        py::arg("weighted_cos").noconvert(),        //
        py::arg("one_over_determinant").noconvert() //
  );
  m.def("repixelize_pol_IQU", numpy_bind_repixelize_pol_IQU<int32_t, double>,
        py::arg("new_npix"),                        //
        py::arg("pixel_mask").noconvert(),          //
        py::arg("weighted_counts").noconvert(),     //
        py::arg("weighted_sin_sq").noconvert(),     //
        py::arg("weighted_cos_sq").noconvert(),     //
        py::arg("weighted_sincos").noconvert(),     //
        py::arg("weighted_sin").noconvert(),        //
        py::arg("weighted_cos").noconvert(),        //
        py::arg("one_over_determinant").noconvert() //
  );
  m.def("repixelize_pol_IQU", numpy_bind_repixelize_pol_IQU<int64_t, double>,
        py::arg("new_npix"),                        //
        py::arg("pixel_mask").noconvert(),          //
        py::arg("weighted_counts").noconvert(),     //
        py::arg("weighted_sin_sq").noconvert(),     //
        py::arg("weighted_cos_sq").noconvert(),     //
        py::arg("weighted_sincos").noconvert(),     //
        py::arg("weighted_sin").noconvert(),        //
        py::arg("weighted_cos").noconvert(),        //
        py::arg("one_over_determinant").noconvert() //
  );
  m.def("flag_bad_pixel_samples", numpy_bind_flag_bad_pixel_samples<int32_t>,
        py::arg("nsamples"),
        py::arg("pixel_flag").noconvert(),    //
        py::arg("old2new_pixel").noconvert(), //
        py::arg("pointings").noconvert(),     //
        py::arg("pointings_flag").noconvert() //
  );
  m.def("flag_bad_pixel_samples", numpy_bind_flag_bad_pixel_samples<int64_t>,
        py::arg("nsamples"),
        py::arg("pixel_flag").noconvert(),    //
        py::arg("old2new_pixel").noconvert(), //
        py::arg("pointings").noconvert(),     //
        py::arg("pointings_flag").noconvert() //
  );
}
