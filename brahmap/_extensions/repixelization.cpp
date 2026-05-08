#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#ifndef _DISABLE_OMP
#include <omp.h>
#endif

namespace nb = nanobind;

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

template <typename dint, typename dfloat, typename device> //
void register_repixelize_pol(nb::module_ &m) {
  using arr_dint = nb::ndarray<dint, nb::ndim<1>, device, nb::c_contig>;
  using arr_dfloat = nb::ndarray<dfloat, nb::ndim<1>, device, nb::c_contig>;

  m.def(
      "repixelize_pol_I",                 //
      [](                                 //
          const ssize_t new_npix,         //
          const arr_dint observed_pixels, //
          arr_dint hit_counts,            //
          arr_dfloat weighted_counts      //
      ) {
        repixelize_pol_I(           //
            new_npix,               //
            observed_pixels.data(), //
            hit_counts.data(),      //
            weighted_counts.data()  //
        );
      },
      nb::arg("new_npix"),                    //
      nb::arg("observed_pixels").noconvert(), //
      nb::arg("hit_counts").noconvert(),      //
      nb::arg("weighted_counts").noconvert()  //
  );

  m.def(
      "repixelize_pol_QU",                //
      [](                                 //
          const ssize_t new_npix,         //
          const arr_dint observed_pixels, //
          arr_dint hit_counts,            //
          arr_dfloat weighted_counts,     //
          arr_dfloat weighted_sin_sq,     //
          arr_dfloat weighted_cos_sq,     //
          arr_dfloat weighted_sincos,     //
          arr_dfloat one_over_determinant //
      ) {
        repixelize_pol_QU(              //
            new_npix,                   //
            observed_pixels.data(),     //
            hit_counts.data(),          //
            weighted_counts.data(),     //
            weighted_sin_sq.data(),     //
            weighted_cos_sq.data(),     //
            weighted_sincos.data(),     //
            one_over_determinant.data() //
        );
      },
      nb::arg("new_npix"),                        //
      nb::arg("observed_pixels").noconvert(),     //
      nb::arg("hit_counts").noconvert(),          //
      nb::arg("weighted_counts").noconvert(),     //
      nb::arg("weighted_sin_sq").noconvert(),     //
      nb::arg("weighted_cos_sq").noconvert(),     //
      nb::arg("weighted_sincos").noconvert(),     //
      nb::arg("one_over_determinant").noconvert() //
  );

  m.def(
      "repixelize_pol_IQU",               //
      [](                                 //
          const ssize_t new_npix,         //
          const arr_dint observed_pixels, //
          arr_dint hit_counts,            //
          arr_dfloat weighted_counts,     //
          arr_dfloat weighted_sin_sq,     //
          arr_dfloat weighted_cos_sq,     //
          arr_dfloat weighted_sincos,     //
          arr_dfloat weighted_sin,        //
          arr_dfloat weighted_cos,        //
          arr_dfloat one_over_determinant //
      ) {
        repixelize_pol_IQU(             //
            new_npix,                   //
            observed_pixels.data(),     //
            hit_counts.data(),          //
            weighted_counts.data(),     //
            weighted_sin_sq.data(),     //
            weighted_cos_sq.data(),     //
            weighted_sincos.data(),     //
            weighted_sin.data(),        //
            weighted_cos.data(),        //
            one_over_determinant.data() //
        );
      },
      nb::arg("new_npix"),                        //
      nb::arg("observed_pixels").noconvert(),     //
      nb::arg("hit_counts").noconvert(),          //
      nb::arg("weighted_counts").noconvert(),     //
      nb::arg("weighted_sin_sq").noconvert(),     //
      nb::arg("weighted_cos_sq").noconvert(),     //
      nb::arg("weighted_sincos").noconvert(),     //
      nb::arg("weighted_sin").noconvert(),        //
      nb::arg("weighted_cos").noconvert(),        //
      nb::arg("one_over_determinant").noconvert() //
  );
}

template <typename dint, typename device> //
void register_flag_bad_pixel_samples(nb::module_ &m) {
  using arr_dint = nb::ndarray<dint, nb::ndim<1>, device, nb::c_contig>;
  using arr_bool = nb::ndarray<bool, nb::ndim<1>, device, nb::c_contig>;

  m.def(
      "flag_bad_pixel_samples",         //
      [](                               //
          const ssize_t nsamples,       //
          const arr_bool pixel_flag,    //
          const arr_dint old2new_pixel, //
          arr_dint pointings,           //
          arr_bool pointings_flag       //
      ) {
        flag_bad_pixel_samples(   //
            nsamples,             //
            pixel_flag.data(),    //
            old2new_pixel.data(), //
            pointings.data(),     //
            pointings_flag.data() //
        );
      },
      nb::arg("nsamples"),                  //
      nb::arg("pixel_flag").noconvert(),    //
      nb::arg("old2new_pixel").noconvert(), //
      nb::arg("pointings").noconvert(),     //
      nb::arg("pointings_flag").noconvert() //
  );
}

NB_MODULE(repixelize, m) {
  m.doc() = "repixelize";

  register_repixelize_pol<int64_t, double, nb::device::cpu>(m);
  register_repixelize_pol<int32_t, double, nb::device::cpu>(m);
  register_repixelize_pol<int64_t, float, nb::device::cpu>(m);
  register_repixelize_pol<int32_t, float, nb::device::cpu>(m);

  register_flag_bad_pixel_samples<int64_t, nb::device::cpu>(m);
  register_flag_bad_pixel_samples<int32_t, nb::device::cpu>(m);
}
