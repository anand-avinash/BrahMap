#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#ifndef _DISABLE_OMP
#include <omp.h>
#endif

namespace py = pybind11;

template <typename dint, typename dfloat>
void repixelize_pol_I(                      //
    const ssize_t new_npix,                 //
    const dint *__restrict observed_pixels, //
    dint *__restrict hit_counts,            //
    dfloat *__restrict weighted_counts      //
) {

  for (ssize_t idx = 0; idx < new_npix; ++idx) {
    dint pixel = observed_pixels[idx];
    hit_counts[idx] = hit_counts[pixel];
    weighted_counts[idx] = weighted_counts[pixel];
  } // for

  return;

} // repixelize_pol_I()

template <typename dint, typename dfloat>
void repixelize_pol_QU(                     //
    const ssize_t new_npix,                 //
    const dint *__restrict observed_pixels, //
    dint *__restrict hit_counts,            //
    dfloat *__restrict weighted_counts,     //
    dfloat *__restrict weighted_sin_sq,     //
    dfloat *__restrict weighted_cos_sq,     //
    dfloat *__restrict weighted_sincos,     //
    dfloat *__restrict one_over_determinant //
) {

  for (ssize_t idx = 0; idx < new_npix; ++idx) {
    dint pixel = observed_pixels[idx];
    hit_counts[idx] = hit_counts[pixel];
    weighted_counts[idx] = weighted_counts[pixel];
    weighted_sin_sq[idx] = weighted_sin_sq[pixel];
    weighted_cos_sq[idx] = weighted_cos_sq[pixel];
    weighted_sincos[idx] = weighted_sincos[pixel];
    one_over_determinant[idx] = 1.0 / one_over_determinant[pixel];
  } // for

  return;

} // repixelize_pol_QU()

template <typename dint, typename dfloat>
void repixelize_pol_IQU(                    //
    const ssize_t new_npix,                 //
    const dint *__restrict observed_pixels, //
    dint *__restrict hit_counts,            //
    dfloat *__restrict weighted_counts,     //
    dfloat *__restrict weighted_sin_sq,     //
    dfloat *__restrict weighted_cos_sq,     //
    dfloat *__restrict weighted_sincos,     //
    dfloat *__restrict weighted_sin,        //
    dfloat *__restrict weighted_cos,        //
    dfloat *__restrict one_over_determinant //
) {

  for (ssize_t idx = 0; idx < new_npix; ++idx) {
    dint pixel = observed_pixels[idx];
    hit_counts[idx] = hit_counts[pixel];
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
void flag_bad_pixel_samples(              //
    const ssize_t nsamples,               //
    const bool *__restrict pixel_flag,    //
    const dint *__restrict old2new_pixel, //
    dint *__restrict pointings,           //
    bool *__restrict pointings_flag       //
) {

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < nsamples; ++idx) {
    dint pixel = pointings[idx];
    bool pixflag = pixel_flag[pixel];
    pointings[idx] = pixflag * old2new_pixel[pixel];
    pointings_flag[idx] &= pixflag;
  } // for

  return;
} //  flag_bad_pixel_samples()

template <template <typename, int = py::array::c_style> class buffer_t,
          typename dint, typename dfloat>
std::function<void(                       //
    const ssize_t new_npix,               //
    const buffer_t<dint> observed_pixels, //
    buffer_t<dint> hit_counts,            //
    buffer_t<dfloat> weighted_counts      //
    )>
    numpy_bind_repixelize_pol_I =        //
    [](const ssize_t new_npix,           //
       const py::buffer observed_pixels, //
       py::buffer hit_counts,            //
       py::buffer weighted_counts        //
    ) {
      py::buffer_info observed_pixels_info = observed_pixels.request();
      py::buffer_info hit_counts_info = hit_counts.request();
      py::buffer_info weighted_counts_info = weighted_counts.request();

      const dint *observed_pixels_ptr =
          reinterpret_cast<const dint *>(observed_pixels_info.ptr);
      dint *hit_counts_ptr = reinterpret_cast<dint *>(hit_counts_info.ptr);
      dfloat *weighted_counts_ptr =
          reinterpret_cast<dfloat *>(weighted_counts_info.ptr);

      repixelize_pol_I(        //
          new_npix,            //
          observed_pixels_ptr, //
          hit_counts_ptr,      //
          weighted_counts_ptr  //
      );
    }; // numpy_bind_repixelize_pol_I()

template <template <typename, int = py::array::c_style> class buffer_t,
          typename dint, typename dfloat>
std::function<void(                       //
    const ssize_t new_npix,               //
    const buffer_t<dint> observed_pixels, //
    buffer_t<dint> hit_counts,            //
    buffer_t<dfloat> weighted_counts,     //
    buffer_t<dfloat> weighted_sin_sq,     //
    buffer_t<dfloat> weighted_cos_sq,     //
    buffer_t<dfloat> weighted_sincos,     //
    buffer_t<dfloat> one_over_determinant //
    )>
    numpy_bind_repixelize_pol_QU =       //
    [](const ssize_t new_npix,           //
       const py::buffer observed_pixels, //
       py::buffer hit_counts,            //
       py::buffer weighted_counts,       //
       py::buffer weighted_sin_sq,       //
       py::buffer weighted_cos_sq,       //
       py::buffer weighted_sincos,       //
       py::buffer one_over_determinant   //
    ) {
      py::buffer_info observed_pixels_info = observed_pixels.request();
      py::buffer_info hit_counts_info = hit_counts.request();
      py::buffer_info weighted_counts_info = weighted_counts.request();
      py::buffer_info weighted_sin_sq_info = weighted_sin_sq.request();
      py::buffer_info weighted_cos_sq_info = weighted_cos_sq.request();
      py::buffer_info weighted_sincos_info = weighted_sincos.request();
      py::buffer_info one_over_determinant_info =
          one_over_determinant.request();

      const dint *observed_pixels_ptr =
          reinterpret_cast<const dint *>(observed_pixels_info.ptr);
      dint *hit_counts_ptr = reinterpret_cast<dint *>(hit_counts_info.ptr);
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
          observed_pixels_ptr,     //
          hit_counts_ptr,          //
          weighted_counts_ptr,     //
          weighted_sin_sq_ptr,     //
          weighted_cos_sq_ptr,     //
          weighted_sincos_ptr,     //
          one_over_determinant_ptr //
      );
    }; // numpy_bind_repixelize_pol_QU()

template <template <typename, int = py::array::c_style> class buffer_t,
          typename dint, typename dfloat>
std::function<void(                       //
    const ssize_t new_npix,               //
    const buffer_t<dint> observed_pixels, //
    buffer_t<dint> hit_counts,            //
    buffer_t<dfloat> weighted_counts,     //
    buffer_t<dfloat> weighted_sin_sq,     //
    buffer_t<dfloat> weighted_cos_sq,     //
    buffer_t<dfloat> weighted_sincos,     //
    buffer_t<dfloat> weighted_sin,        //
    buffer_t<dfloat> weighted_cos,        //
    buffer_t<dfloat> one_over_determinant //
    )>
    numpy_bind_repixelize_pol_IQU =      //
    [](const ssize_t new_npix,           //
       const py::buffer observed_pixels, //
       py::buffer hit_counts,            //
       py::buffer weighted_counts,       //
       py::buffer weighted_sin_sq,       //
       py::buffer weighted_cos_sq,       //
       py::buffer weighted_sincos,       //
       py::buffer weighted_sin,          //
       py::buffer weighted_cos,          //
       py::buffer one_over_determinant   //
    ) {
      py::buffer_info observed_pixels_info = observed_pixels.request();
      py::buffer_info hit_counts_info = hit_counts.request();
      py::buffer_info weighted_counts_info = weighted_counts.request();
      py::buffer_info weighted_sin_sq_info = weighted_sin_sq.request();
      py::buffer_info weighted_cos_sq_info = weighted_cos_sq.request();
      py::buffer_info weighted_sincos_info = weighted_sincos.request();
      py::buffer_info weighted_sin_info = weighted_sin.request();
      py::buffer_info weighted_cos_info = weighted_cos.request();
      py::buffer_info one_over_determinant_info =
          one_over_determinant.request();

      const dint *observed_pixels_ptr =
          reinterpret_cast<const dint *>(observed_pixels_info.ptr);
      dint *hit_counts_ptr = reinterpret_cast<dint *>(hit_counts_info.ptr);
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
          observed_pixels_ptr,     //
          hit_counts_ptr,          //
          weighted_counts_ptr,     //
          weighted_sin_sq_ptr,     //
          weighted_cos_sq_ptr,     //
          weighted_sincos_ptr,     //
          weighted_sin_ptr,        //
          weighted_cos_ptr,        //
          one_over_determinant_ptr //
      );
    }; // numpy_bind_repixelize_pol_IQU()

template <template <typename, int = py::array::c_style> class buffer_t,
          typename dint>
std::function<void(                     //
    const ssize_t nsamples,             //
    const buffer_t<bool> pixel_flag,    //
    const buffer_t<dint> old2new_pixel, //
    buffer_t<dint> pointings,           //
    buffer_t<bool> poitnings_flag       //
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
  m.def("repixelize_pol_I",
        numpy_bind_repixelize_pol_I<py::array_t, int32_t, float>,
        py::arg("new_npix"),                    //
        py::arg("observed_pixels").noconvert(), //
        py::arg("hit_counts").noconvert(),      //
        py::arg("weighted_counts").noconvert()  //
  );
  m.def("repixelize_pol_I",
        numpy_bind_repixelize_pol_I<py::array_t, int64_t, float>,
        py::arg("new_npix"),                    //
        py::arg("observed_pixels").noconvert(), //
        py::arg("hit_counts").noconvert(),      //
        py::arg("weighted_counts").noconvert()  //
  );
  m.def("repixelize_pol_I",
        numpy_bind_repixelize_pol_I<py::array_t, int32_t, double>,
        py::arg("new_npix"),                    //
        py::arg("observed_pixels").noconvert(), //
        py::arg("hit_counts").noconvert(),      //
        py::arg("weighted_counts").noconvert()  //
  );
  m.def("repixelize_pol_I",
        numpy_bind_repixelize_pol_I<py::array_t, int64_t, double>,
        py::arg("new_npix"),                    //
        py::arg("observed_pixels").noconvert(), //
        py::arg("hit_counts").noconvert(),      //
        py::arg("weighted_counts").noconvert()  //
  );

  m.def("repixelize_pol_QU",
        numpy_bind_repixelize_pol_QU<py::array_t, int32_t, float>,
        py::arg("new_npix"),                        //
        py::arg("observed_pixels").noconvert(),     //
        py::arg("hit_counts").noconvert(),          //
        py::arg("weighted_counts").noconvert(),     //
        py::arg("weighted_sin_sq").noconvert(),     //
        py::arg("weighted_cos_sq").noconvert(),     //
        py::arg("weighted_sincos").noconvert(),     //
        py::arg("one_over_determinant").noconvert() //
  );
  m.def("repixelize_pol_QU",
        numpy_bind_repixelize_pol_QU<py::array_t, int64_t, float>,
        py::arg("new_npix"),                        //
        py::arg("observed_pixels").noconvert(),     //
        py::arg("hit_counts").noconvert(),          //
        py::arg("weighted_counts").noconvert(),     //
        py::arg("weighted_sin_sq").noconvert(),     //
        py::arg("weighted_cos_sq").noconvert(),     //
        py::arg("weighted_sincos").noconvert(),     //
        py::arg("one_over_determinant").noconvert() //
  );
  m.def("repixelize_pol_QU",
        numpy_bind_repixelize_pol_QU<py::array_t, int32_t, double>,
        py::arg("new_npix"),                        //
        py::arg("observed_pixels").noconvert(),     //
        py::arg("hit_counts").noconvert(),          //
        py::arg("weighted_counts").noconvert(),     //
        py::arg("weighted_sin_sq").noconvert(),     //
        py::arg("weighted_cos_sq").noconvert(),     //
        py::arg("weighted_sincos").noconvert(),     //
        py::arg("one_over_determinant").noconvert() //
  );
  m.def("repixelize_pol_QU",
        numpy_bind_repixelize_pol_QU<py::array_t, int64_t, double>,
        py::arg("new_npix"),                        //
        py::arg("observed_pixels").noconvert(),     //
        py::arg("hit_counts").noconvert(),          //
        py::arg("weighted_counts").noconvert(),     //
        py::arg("weighted_sin_sq").noconvert(),     //
        py::arg("weighted_cos_sq").noconvert(),     //
        py::arg("weighted_sincos").noconvert(),     //
        py::arg("one_over_determinant").noconvert() //
  );

  m.def("repixelize_pol_IQU",
        numpy_bind_repixelize_pol_IQU<py::array_t, int32_t, float>,
        py::arg("new_npix"),                        //
        py::arg("observed_pixels").noconvert(),     //
        py::arg("hit_counts").noconvert(),          //
        py::arg("weighted_counts").noconvert(),     //
        py::arg("weighted_sin_sq").noconvert(),     //
        py::arg("weighted_cos_sq").noconvert(),     //
        py::arg("weighted_sincos").noconvert(),     //
        py::arg("weighted_sin").noconvert(),        //
        py::arg("weighted_cos").noconvert(),        //
        py::arg("one_over_determinant").noconvert() //
  );
  m.def("repixelize_pol_IQU",
        numpy_bind_repixelize_pol_IQU<py::array_t, int64_t, float>,
        py::arg("new_npix"),                        //
        py::arg("observed_pixels").noconvert(),     //
        py::arg("hit_counts").noconvert(),          //
        py::arg("weighted_counts").noconvert(),     //
        py::arg("weighted_sin_sq").noconvert(),     //
        py::arg("weighted_cos_sq").noconvert(),     //
        py::arg("weighted_sincos").noconvert(),     //
        py::arg("weighted_sin").noconvert(),        //
        py::arg("weighted_cos").noconvert(),        //
        py::arg("one_over_determinant").noconvert() //
  );
  m.def("repixelize_pol_IQU",
        numpy_bind_repixelize_pol_IQU<py::array_t, int32_t, double>,
        py::arg("new_npix"),                        //
        py::arg("observed_pixels").noconvert(),     //
        py::arg("hit_counts").noconvert(),          //
        py::arg("weighted_counts").noconvert(),     //
        py::arg("weighted_sin_sq").noconvert(),     //
        py::arg("weighted_cos_sq").noconvert(),     //
        py::arg("weighted_sincos").noconvert(),     //
        py::arg("weighted_sin").noconvert(),        //
        py::arg("weighted_cos").noconvert(),        //
        py::arg("one_over_determinant").noconvert() //
  );
  m.def("repixelize_pol_IQU",
        numpy_bind_repixelize_pol_IQU<py::array_t, int64_t, double>,
        py::arg("new_npix"),                        //
        py::arg("observed_pixels").noconvert(),     //
        py::arg("hit_counts").noconvert(),          //
        py::arg("weighted_counts").noconvert(),     //
        py::arg("weighted_sin_sq").noconvert(),     //
        py::arg("weighted_cos_sq").noconvert(),     //
        py::arg("weighted_sincos").noconvert(),     //
        py::arg("weighted_sin").noconvert(),        //
        py::arg("weighted_cos").noconvert(),        //
        py::arg("one_over_determinant").noconvert() //
  );
  m.def("flag_bad_pixel_samples",
        numpy_bind_flag_bad_pixel_samples<py::array_t, int32_t>,
        py::arg("nsamples"),
        py::arg("pixel_flag").noconvert(),    //
        py::arg("old2new_pixel").noconvert(), //
        py::arg("pointings").noconvert(),     //
        py::arg("pointings_flag").noconvert() //
  );
  m.def("flag_bad_pixel_samples",
        numpy_bind_flag_bad_pixel_samples<py::array_t, int64_t>,
        py::arg("nsamples"),
        py::arg("pixel_flag").noconvert(),    //
        py::arg("old2new_pixel").noconvert(), //
        py::arg("pointings").noconvert(),     //
        py::arg("pointings_flag").noconvert() //
  );
}
