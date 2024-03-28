#include <cmath>
#include <functional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename dint, typename dfloat>
dint compute_weights_pol_I(      //
    const ssize_t npix,          //
    const ssize_t nsamples,      //
    const dint *pointings,       //
    const bool *pointings_flag,  //
    const dfloat *noise_weights, //
    dfloat *weighted_counts,     //
    dint *pixel_mask             //
) {

  for (ssize_t idx = 0; idx < nsamples; ++idx) {
    ssize_t pixel = pointings[idx];
    dfloat weight = pointings_flag[idx] * noise_weights[idx];

    weighted_counts[pixel] += weight;
  } // for

  dint new_npix = 0;
  for (ssize_t idx = 0; idx < npix; ++idx) {
    if (weighted_counts[idx] > 0) {
      pixel_mask[new_npix] = idx;
      ++new_npix;
    } // if
  }   // for

  return new_npix;

} // compute_weights_pol_I()

template <typename dint, typename dfloat>
void compute_weights_pol_QU(     //
    const ssize_t nsamples,      //
    const dint *pointings,       //
    const bool *pointings_flag,  //
    const dfloat *noise_weights, //
    const dfloat *pol_angles,    //
    dfloat *weighted_counts,     //
    dfloat *sin2phi,             //
    dfloat *cos2phi,             //
    dfloat *weighted_sin_sq,     //
    dfloat *weighted_cos_sq,     //
    dfloat *weighted_sincos      //

) {

  for (ssize_t idx = 0; idx < nsamples; ++idx) {
    dfloat angle = pol_angles[idx];
    sin2phi[idx] = std::sin(2.0 * angle);
    cos2phi[idx] = std::cos(2.0 * angle);
  } // for

  for (ssize_t idx = 0; idx < nsamples; ++idx) {
    ssize_t pixel = pointings[idx];
    dfloat weight = pointings_flag[idx] * noise_weights[idx];

    weighted_counts[pixel] += weight;
    weighted_sin_sq[pixel] += weight * sin2phi[idx] * sin2phi[idx];
    weighted_cos_sq[pixel] += weight * cos2phi[idx] * cos2phi[idx];
    weighted_sincos[pixel] += weight * sin2phi[idx] * cos2phi[idx];
  } // for

  return;

} // compute_weights_pol_QU()

template <typename dint, typename dfloat>
void compute_weights_pol_IQU(    //
    const ssize_t nsamples,      //
    const dint *pointings,       //
    const bool *pointings_flag,  //
    const dfloat *noise_weights, //
    const dfloat *pol_angles,    //
    dfloat *weighted_counts,     //
    dfloat *sin2phi,             //
    dfloat *cos2phi,             //
    dfloat *weighted_sin_sq,     //
    dfloat *weighted_cos_sq,     //
    dfloat *weighted_sincos,     //
    dfloat *weighted_sin,        //
    dfloat *weighted_cos         //

) {

  for (ssize_t idx = 0; idx < nsamples; ++idx) {
    dfloat angle = pol_angles[idx];
    sin2phi[idx] = std::sin(2.0 * angle);
    cos2phi[idx] = std::cos(2.0 * angle);
  } // for

  for (ssize_t idx = 0; idx < nsamples; ++idx) {
    ssize_t pixel = pointings[idx];
    dfloat weight = pointings_flag[idx] * noise_weights[idx];

    weighted_counts[pixel] += weight;
    weighted_sin[pixel] += weight * sin2phi[idx];
    weighted_sin_sq[pixel] += weight * sin2phi[idx] * sin2phi[idx];
    weighted_cos[pixel] += weight * cos2phi[idx];
    weighted_cos_sq[pixel] += weight * cos2phi[idx] * cos2phi[idx];
    weighted_sincos[pixel] += weight * sin2phi[idx] * cos2phi[idx];
  } // for

  return;

} // compute_weights_pol_IQU()

template <typename dint, typename dfloat>
dint get_pixel_mask_pol(           //
    const int solver_type,         //
    const ssize_t npix,            //
    const dfloat threshold,        //
    const dfloat *weighted_counts, //
    const dfloat *weighted_sin_sq, //
    const dfloat *weighted_cos_sq, //
    const dfloat *weighted_sincos, //
    dint *pixel_mask               //
) {

  dfloat weight_threshold = (dfloat)(solver_type - 1);

  dint new_npix = 0;
  for (ssize_t idx = 0; idx < npix; ++idx) {
    dfloat determinant = weighted_sin_sq[idx] * weighted_cos_sq[idx] -
                         weighted_sincos[idx] * weighted_sincos[idx];
    dfloat trace = weighted_cos_sq[idx] + weighted_sin_sq[idx];
    dfloat dsqrt = std::sqrt(trace * trace / 4.0 - determinant);
    dfloat lambda_max = trace / 2.0 + dsqrt;
    dfloat lambda_min = trace / 2.0 - dsqrt;
    dfloat cond_num = std::abs(lambda_max / lambda_min);

    dfloat weight = weighted_counts[idx];

    if ((cond_num <= threshold) && (weight > weight_threshold)) {
      pixel_mask[new_npix] = idx;
      ++new_npix;
    } // if

  } // for

  return new_npix;

} // get_pixel_mask_pol()

template <typename dint, typename dfloat>
std::function<dint(                          //
    const ssize_t npix,                      //
    const ssize_t nsamples,                  //
    const py::array_t<dint> pointings,       //
    const py::array_t<bool> pointings_flag,  //
    const py::array_t<dfloat> noise_weights, //
    py::array_t<dfloat> weighted_counts,     //
    py::array_t<dint> pixel_mask             //
    )>
    numpy_bind_compute_weights_pol_I =  //
    [](const ssize_t npix,              //
       const ssize_t nsamples,          //
       const py::buffer pointings,      //
       const py::buffer pointings_flag, //
       const py::buffer noise_weights,  //
       py::buffer weighted_counts,      //
       py::buffer pixel_mask            //
       ) -> dint {
  py::buffer_info pointings_info = pointings.request();
  py::buffer_info pointings_flags_info = pointings_flag.request();
  py::buffer_info noise_weights_info = noise_weights.request();
  py::buffer_info weighted_counts_info = weighted_counts.request();
  py::buffer_info pixel_mask_info = pixel_mask.request();

  const dint *pointings_ptr =
      reinterpret_cast<const dint *>(pointings_info.ptr);
  const bool *pointings_flags_ptr =
      reinterpret_cast<const bool *>(pointings_flags_info.ptr);
  const dfloat *noise_weights_ptr =
      reinterpret_cast<const dfloat *>(noise_weights_info.ptr);
  dfloat *weighted_counts_ptr =
      reinterpret_cast<dfloat *>(weighted_counts_info.ptr);
  dint *pixel_mask_ptr = reinterpret_cast<dint *>(pixel_mask_info.ptr);

  dint new_npix = compute_weights_pol_I( //
      npix,                              //
      nsamples,                          //
      pointings_ptr,                     //
      pointings_flags_ptr,               //
      noise_weights_ptr,                 //
      weighted_counts_ptr,               //
      pixel_mask_ptr                     //
  );

  return new_npix;
}; // numpy_bind_compute_weights_pol_I

template <typename dint, typename dfloat>
std::function<void(                          //
    const ssize_t nsamples,                  //
    const py::array_t<dint> pointings,       //
    const py::array_t<bool> pointings_flag,  //
    const py::array_t<dfloat> noise_weights, //
    const py::array_t<dfloat> pol_angles,    //
    py::array_t<dfloat> weighted_counts,     //
    py::array_t<dfloat> sin2phi,             //
    py::array_t<dfloat> cos2phi,             //
    py::array_t<dfloat> weighted_sin_sq,     //
    py::array_t<dfloat> weighted_cos_sq,     //
    py::array_t<dfloat> weighted_sincos      //
    )>
    numpy_bind_compute_weights_pol_QU = //
    [](const ssize_t nsamples,          //
       const py::buffer pointings,      //
       const py::buffer pointings_flag, //
       const py::buffer noise_weights,  //
       const py::buffer pol_angles,     //
       py::buffer weighted_counts,      //
       py::buffer sin2phi,              //
       py::buffer cos2phi,              //
       py::buffer weighted_sin_sq,      //
       py::buffer weighted_cos_sq,      //
       py::buffer weighted_sincos       //
    ) {
      py::buffer_info pointings_info = pointings.request();
      py::buffer_info pointings_flag_info = pointings_flag.request();
      py::buffer_info noise_weights_info = noise_weights.request();
      py::buffer_info pol_angles_info = pol_angles.request();
      py::buffer_info weighted_counts_info = weighted_counts.request();
      py::buffer_info sin2phi_info = sin2phi.request();
      py::buffer_info cos2phi_info = cos2phi.request();
      py::buffer_info weighted_sin_sq_info = weighted_sin_sq.request();
      py::buffer_info weighted_cos_sq_info = weighted_cos_sq.request();
      py::buffer_info weighted_sincos_info = weighted_sincos.request();

      const dint *pointings_ptr =
          reinterpret_cast<const dint *>(pointings_info.ptr);
      const bool *pointings_flag_ptr =
          reinterpret_cast<const bool *>(pointings_flag_info.ptr);
      const dfloat *noise_weights_ptr =
          reinterpret_cast<const dfloat *>(noise_weights_info.ptr);
      const dfloat *pol_angles_ptr =
          reinterpret_cast<const dfloat *>(pol_angles_info.ptr);
      dfloat *weighted_counts_ptr =
          reinterpret_cast<dfloat *>(weighted_counts_info.ptr);
      dfloat *sin2phi_ptr = reinterpret_cast<dfloat *>(sin2phi_info.ptr);
      dfloat *cos2phi_ptr = reinterpret_cast<dfloat *>(cos2phi_info.ptr);
      dfloat *weighted_sin_sq_ptr =
          reinterpret_cast<dfloat *>(weighted_sin_sq_info.ptr);
      dfloat *weighted_cos_sq_ptr =
          reinterpret_cast<dfloat *>(weighted_cos_sq_info.ptr);
      dfloat *weighted_sincos_ptr =
          reinterpret_cast<dfloat *>(weighted_sincos_info.ptr);

      compute_weights_pol_QU(  //
          nsamples,            //
          pointings_ptr,       //
          pointings_flag_ptr,  //
          noise_weights_ptr,   //
          pol_angles_ptr,      //
          weighted_counts_ptr, //
          sin2phi_ptr,         //
          cos2phi_ptr,         //
          weighted_sin_sq_ptr, //
          weighted_cos_sq_ptr, //
          weighted_sincos_ptr  //
      );

      return;
    }; // numpy_bind_compute_weights_pol_QU()

template <typename dint, typename dfloat>
std::function<void(                          //
    const ssize_t nsamples,                  //
    const py::array_t<dint> pointings,       //
    const py::array_t<bool> pointings_flag,  //
    const py::array_t<dfloat> noise_weights, //
    const py::array_t<dfloat> pol_angles,    //
    py::array_t<dfloat> weighted_counts,     //
    py::array_t<dfloat> sin2phi,             //
    py::array_t<dfloat> cos2phi,             //
    py::array_t<dfloat> weighted_sin_sq,     //
    py::array_t<dfloat> weighted_cos_sq,     //
    py::array_t<dfloat> weighted_sincos,     //
    py::array_t<dfloat> weighted_sin,        //
    py::array_t<dfloat> weighted_cos         //
    )>
    numpy_bind_compute_weights_pol_IQU = //
    [](const ssize_t nsamples,           //
       const py::buffer pointings,       //
       const py::buffer pointings_flag,  //
       const py::buffer noise_weights,   //
       const py::buffer pol_angles,      //
       py::buffer weighted_counts,       //
       py::buffer sin2phi,               //
       py::buffer cos2phi,               //
       py::buffer weighted_sin_sq,       //
       py::buffer weighted_cos_sq,       //
       py::buffer weighted_sincos,       //
       py::buffer weighted_sin,          //
       py::buffer weighted_cos           //
    ) {
      py::buffer_info pointings_info = pointings.request();
      py::buffer_info pointings_flag_info = pointings_flag.request();
      py::buffer_info noise_weights_info = noise_weights.request();
      py::buffer_info pol_angles_info = pol_angles.request();
      py::buffer_info weighted_counts_info = weighted_counts.request();
      py::buffer_info sin2phi_info = sin2phi.request();
      py::buffer_info cos2phi_info = cos2phi.request();
      py::buffer_info weighted_sin_sq_info = weighted_sin_sq.request();
      py::buffer_info weighted_cos_sq_info = weighted_cos_sq.request();
      py::buffer_info weighted_sincos_info = weighted_sincos.request();
      py::buffer_info weighted_sin_info = weighted_sin.request();
      py::buffer_info weighted_cos_info = weighted_cos.request();

      const dint *pointings_ptr =
          reinterpret_cast<const dint *>(pointings_info.ptr);
      const bool *pointings_flag_ptr =
          reinterpret_cast<const bool *>(pointings_flag_info.ptr);
      const dfloat *noise_weights_ptr =
          reinterpret_cast<const dfloat *>(noise_weights_info.ptr);
      const dfloat *pol_angles_ptr =
          reinterpret_cast<const dfloat *>(pol_angles_info.ptr);
      dfloat *weighted_counts_ptr =
          reinterpret_cast<dfloat *>(weighted_counts_info.ptr);
      dfloat *sin2phi_ptr = reinterpret_cast<dfloat *>(sin2phi_info.ptr);
      dfloat *cos2phi_ptr = reinterpret_cast<dfloat *>(cos2phi_info.ptr);
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

      compute_weights_pol_IQU( //
          nsamples,            //
          pointings_ptr,       //
          pointings_flag_ptr,  //
          noise_weights_ptr,   //
          pol_angles_ptr,      //
          weighted_counts_ptr, //
          sin2phi_ptr,         //
          cos2phi_ptr,         //
          weighted_sin_sq_ptr, //
          weighted_cos_sq_ptr, //
          weighted_sincos_ptr, //
          weighted_sin_ptr,    //
          weighted_cos_ptr     //
      );

      return;
    }; // numpy_bind_compute_weights_pol_IQU()

template <typename dint, typename dfloat>
std::function<dint(                            //
    const int solver_type,                     //
    const ssize_t npix,                        //
    const dfloat threshold,                    //
    const py::array_t<dfloat> weighted_counts, //
    const py::array_t<dfloat> weighted_sin_sq, //
    const py::array_t<dfloat> weighted_cos_sq, //
    const py::array_t<dfloat> weighted_sincos, //
    py::array_t<dint> pixel_mask               //
    )>
    numpy_bind_get_pixel_mask_pol =      //
    [](const int solver_type,            //
       const ssize_t npix,               //
       const dfloat threshold,           //
       const py::buffer weighted_counts, //
       const py::buffer weighted_sin_sq, //
       const py::buffer weighted_cos_sq, //
       const py::buffer weighted_sincos, //
       py::buffer pixel_mask             //
       ) -> dint {
  py::buffer_info weighted_counts_info = weighted_counts.request();
  py::buffer_info weighted_sin_sq_info = weighted_sin_sq.request();
  py::buffer_info weighted_cos_sq_info = weighted_cos_sq.request();
  py::buffer_info weighted_sincos_info = weighted_sincos.request();
  py::buffer_info pixel_mask_info = pixel_mask.request();

  const dfloat *weighted_counts_ptr =
      reinterpret_cast<const dfloat *>(weighted_counts_info.ptr);
  const dfloat *weighted_sin_sq_ptr =
      reinterpret_cast<const dfloat *>(weighted_sin_sq_info.ptr);
  const dfloat *weighted_cos_sq_ptr =
      reinterpret_cast<const dfloat *>(weighted_cos_sq_info.ptr);
  const dfloat *weighted_sincos_ptr =
      reinterpret_cast<const dfloat *>(weighted_sincos_info.ptr);
  dint *pixel_mask_ptr = reinterpret_cast<dint *>(pixel_mask_info.ptr);

  dint new_npix = get_pixel_mask_pol( //
      solver_type,                    //
      npix,                           //
      threshold,                      //
      weighted_counts_ptr,            //
      weighted_sin_sq_ptr,            //
      weighted_cos_sq_ptr,            //
      weighted_sincos_ptr,            //
      pixel_mask_ptr                  //
  );

  return new_npix;
}; // numpy_bind_get_pixel_mask_pol()

PYBIND11_MODULE(compute_weights, m) {
  m.doc() = "compute_weights";
  m.def("compute_weights_pol_I",
        numpy_bind_compute_weights_pol_I<int32_t, float>,
        py::arg("npix"),                        //
        py::arg("nsamples"),                    //
        py::arg("pointings").noconvert(),       //
        py::arg("pointings_flag").noconvert(),  //
        py::arg("noise_weights").noconvert(),   //
        py::arg("weighted_counts").noconvert(), //
        py::arg("pixel_mask").noconvert()       //
  );
  m.def("compute_weights_pol_I",
        numpy_bind_compute_weights_pol_I<int64_t, float>,
        py::arg("npix"),                        //
        py::arg("nsamples"),                    //
        py::arg("pointings").noconvert(),       //
        py::arg("pointings_flag").noconvert(),  //
        py::arg("noise_weights").noconvert(),   //
        py::arg("weighted_counts").noconvert(), //
        py::arg("pixel_mask").noconvert()       //
  );
  m.def("compute_weights_pol_I",
        numpy_bind_compute_weights_pol_I<int32_t, double>,
        py::arg("npix"),                        //
        py::arg("nsamples"),                    //
        py::arg("pointings").noconvert(),       //
        py::arg("pointings_flag").noconvert(),  //
        py::arg("noise_weights").noconvert(),   //
        py::arg("weighted_counts").noconvert(), //
        py::arg("pixel_mask").noconvert()       //
  );
  m.def("compute_weights_pol_I",
        numpy_bind_compute_weights_pol_I<int64_t, double>,
        py::arg("npix"),                        //
        py::arg("nsamples"),                    //
        py::arg("pointings").noconvert(),       //
        py::arg("pointings_flag").noconvert(),  //
        py::arg("noise_weights").noconvert(),   //
        py::arg("weighted_counts").noconvert(), //
        py::arg("pixel_mask").noconvert()       //
  );

  m.def("compute_weights_pol_QU",
        numpy_bind_compute_weights_pol_QU<int32_t, float>,
        py::arg("nsamples"),                    //
        py::arg("pointings").noconvert(),       //
        py::arg("pointings_flag").noconvert(),  //
        py::arg("noise_weights").noconvert(),   //
        py::arg("pol_angles").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("sin2phi").noconvert(),         //
        py::arg("cos2phi").noconvert(),         //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert()  //
  );

  m.def("compute_weights_pol_QU",
        numpy_bind_compute_weights_pol_QU<int64_t, float>,
        py::arg("nsamples"),                    //
        py::arg("pointings").noconvert(),       //
        py::arg("pointings_flag").noconvert(),  //
        py::arg("noise_weights").noconvert(),   //
        py::arg("pol_angles").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("sin2phi").noconvert(),         //
        py::arg("cos2phi").noconvert(),         //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert()  //
  );

  m.def("compute_weights_pol_QU",
        numpy_bind_compute_weights_pol_QU<int32_t, double>,
        py::arg("nsamples"),                    //
        py::arg("pointings").noconvert(),       //
        py::arg("pointings_flag").noconvert(),  //
        py::arg("noise_weights").noconvert(),   //
        py::arg("pol_angles").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("sin2phi").noconvert(),         //
        py::arg("cos2phi").noconvert(),         //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert()  //
  );

  m.def("compute_weights_pol_QU",
        numpy_bind_compute_weights_pol_QU<int64_t, double>,
        py::arg("nsamples"),                    //
        py::arg("pointings").noconvert(),       //
        py::arg("pointings_flag").noconvert(),  //
        py::arg("noise_weights").noconvert(),   //
        py::arg("pol_angles").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("sin2phi").noconvert(),         //
        py::arg("cos2phi").noconvert(),         //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert()  //
  );

  m.def("compute_weights_pol_IQU",
        numpy_bind_compute_weights_pol_IQU<int32_t, float>,
        py::arg("nsamples"),                    //
        py::arg("pointings").noconvert(),       //
        py::arg("pointings_flag").noconvert(),  //
        py::arg("noise_weights").noconvert(),   //
        py::arg("pol_angles").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("sin2phi").noconvert(),         //
        py::arg("cos2phi").noconvert(),         //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert(), //
        py::arg("weighted_sin").noconvert(),    //
        py::arg("weighted_cos").noconvert()     //
  );

  m.def("compute_weights_pol_IQU",
        numpy_bind_compute_weights_pol_IQU<int64_t, float>,
        py::arg("nsamples"),                    //
        py::arg("pointings").noconvert(),       //
        py::arg("pointings_flag").noconvert(),  //
        py::arg("noise_weights").noconvert(),   //
        py::arg("pol_angles").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("sin2phi").noconvert(),         //
        py::arg("cos2phi").noconvert(),         //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert(), //
        py::arg("weighted_sin").noconvert(),    //
        py::arg("weighted_cos").noconvert()     //
  );

  m.def("compute_weights_pol_IQU",
        numpy_bind_compute_weights_pol_IQU<int32_t, double>,
        py::arg("nsamples"),                    //
        py::arg("pointings").noconvert(),       //
        py::arg("pointings_flag").noconvert(),  //
        py::arg("noise_weights").noconvert(),   //
        py::arg("pol_angles").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("sin2phi").noconvert(),         //
        py::arg("cos2phi").noconvert(),         //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert(), //
        py::arg("weighted_sin").noconvert(),    //
        py::arg("weighted_cos").noconvert()     //
  );

  m.def("compute_weights_pol_IQU",
        numpy_bind_compute_weights_pol_IQU<int64_t, double>,
        py::arg("nsamples"),                    //
        py::arg("pointings").noconvert(),       //
        py::arg("pointings_flag").noconvert(),  //
        py::arg("noise_weights").noconvert(),   //
        py::arg("pol_angles").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("sin2phi").noconvert(),         //
        py::arg("cos2phi").noconvert(),         //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert(), //
        py::arg("weighted_sin").noconvert(),    //
        py::arg("weighted_cos").noconvert()     //
  );

  m.def("get_pixel_mask_pol", numpy_bind_get_pixel_mask_pol<int32_t, float>,
        py::arg("solver_type"),                 //
        py::arg("npix"),                        //
        py::arg("threshold"),                   //
        py::arg("weighted_counts").noconvert(), //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert(), //
        py::arg("pixel_mask").noconvert()       //
  );
  m.def("get_pixel_mask_pol", numpy_bind_get_pixel_mask_pol<int64_t, float>,
        py::arg("solver_type"),                 //
        py::arg("npix"),                        //
        py::arg("threshold"),                   //
        py::arg("weighted_counts").noconvert(), //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert(), //
        py::arg("pixel_mask").noconvert()       //
  );
  m.def("get_pixel_mask_pol", numpy_bind_get_pixel_mask_pol<int32_t, double>,
        py::arg("solver_type"),                 //
        py::arg("npix"),                        //
        py::arg("threshold"),                   //
        py::arg("weighted_counts").noconvert(), //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert(), //
        py::arg("pixel_mask").noconvert()       //
  );
  m.def("get_pixel_mask_pol", numpy_bind_get_pixel_mask_pol<int64_t, double>,
        py::arg("solver_type"),                 //
        py::arg("npix"),                        //
        py::arg("threshold"),                   //
        py::arg("weighted_counts").noconvert(), //
        py::arg("weighted_sin_sq").noconvert(), //
        py::arg("weighted_cos_sq").noconvert(), //
        py::arg("weighted_sincos").noconvert(), //
        py::arg("pixel_mask").noconvert()       //
  );
}
