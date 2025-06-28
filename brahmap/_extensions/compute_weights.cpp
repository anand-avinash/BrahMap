#include <cmath>
#include <functional>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#ifndef _DISABLE_OMP
#include <omp.h>
#endif

#include "mpi_utils.hpp"

namespace py = pybind11;

template <typename dint, typename dfloat>
dint compute_weights_pol_I(                 //
    const ssize_t npix,                     //
    const ssize_t nsamples,                 //
    const dint *__restrict pointings,       //
    const bool *__restrict pointings_flag,  //
    const dfloat *__restrict noise_weights, //
    dint *__restrict hit_counts,            //
    dfloat *__restrict weighted_counts,     //
    dint *__restrict observed_pixels,       //
    dint *__restrict __old2new_pixel,       //
    bool *__restrict pixel_flag,            //
    const MPI_Comm comm                     //
) {

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < nsamples; ++idx) {
    ssize_t pixel = pointings[idx];
    dfloat weight = pointings_flag[idx] * noise_weights[idx];

#pragma omp atomic update
    hit_counts[pixel] += pointings_flag[idx];
#pragma omp atomic update
    weighted_counts[pixel] += weight;

  } // for

  MPI_Allreduce(MPI_IN_PLACE, hit_counts, npix, mpi_get_type<dint>(), MPI_SUM,
                comm);
  MPI_Allreduce(MPI_IN_PLACE, weighted_counts, npix, mpi_get_type<dfloat>(),
                MPI_SUM, comm);

  dint new_npix = 0;
  for (ssize_t idx = 0; idx < npix; ++idx) {
    if (hit_counts[idx] > 0) {
      observed_pixels[new_npix] = idx;
      __old2new_pixel[idx] = new_npix;
      pixel_flag[idx] = true;
      ++new_npix;
    } // if
  }   // for

  return new_npix;

} // compute_weights_pol_I()

template <typename dint, typename dfloat>
void compute_weights_pol_QU(                 //
    const ssize_t npix,                      //
    const ssize_t nsamples,                  //
    const dint *__restrict pointings,        //
    const bool *__restrict pointings_flag,   //
    const dfloat *__restrict noise_weights,  //
    const dfloat *__restrict pol_angles,     //
    dint *__restrict hit_counts,             //
    dfloat *__restrict weighted_counts,      //
    dfloat *__restrict sin2phi,              //
    dfloat *__restrict cos2phi,              //
    dfloat *__restrict weighted_sin_sq,      //
    dfloat *__restrict weighted_cos_sq,      //
    dfloat *__restrict weighted_sincos,      //
    dfloat *__restrict one_over_determinant, //
    const MPI_Comm comm                      //
) {

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < nsamples; ++idx) {
    dfloat angle = pol_angles[idx];
    sin2phi[idx] = std::sin(2.0 * angle);
    cos2phi[idx] = std::cos(2.0 * angle);
  } // for

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < nsamples; ++idx) {
    ssize_t pixel = pointings[idx];
    dfloat weight = pointings_flag[idx] * noise_weights[idx];

    dfloat wsin_sq = weight * sin2phi[idx] * sin2phi[idx];
    dfloat wcos_sq = weight * cos2phi[idx] * cos2phi[idx];
    dfloat wsincos = weight * sin2phi[idx] * cos2phi[idx];

#pragma omp atomic update
    hit_counts[pixel] += pointings_flag[idx];
#pragma omp atomic update
    weighted_counts[pixel] += weight;
#pragma omp atomic update
    weighted_sin_sq[pixel] += wsin_sq;
#pragma omp atomic update
    weighted_cos_sq[pixel] += wcos_sq;
#pragma omp atomic update
    weighted_sincos[pixel] += wsincos;

  } // for

  MPI_Allreduce(MPI_IN_PLACE, hit_counts, npix, mpi_get_type<dint>(), MPI_SUM,
                comm);
  MPI_Allreduce(MPI_IN_PLACE, weighted_counts, npix, mpi_get_type<dfloat>(),
                MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, weighted_sin_sq, npix, mpi_get_type<dfloat>(),
                MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, weighted_cos_sq, npix, mpi_get_type<dfloat>(),
                MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, weighted_sincos, npix, mpi_get_type<dfloat>(),
                MPI_SUM, comm);

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < npix; ++idx) {
    dfloat determinant = weighted_sin_sq[idx] * weighted_cos_sq[idx] -
                         weighted_sincos[idx] * weighted_sincos[idx];

    one_over_determinant[idx] = determinant;
  } // for

  return;

} // compute_weights_pol_QU()

template <typename dint, typename dfloat>
void compute_weights_pol_IQU(                //
    const ssize_t npix,                      //
    const ssize_t nsamples,                  //
    const dint *__restrict pointings,        //
    const bool *__restrict pointings_flag,   //
    const dfloat *__restrict noise_weights,  //
    const dfloat *__restrict pol_angles,     //
    dint *__restrict hit_counts,             //
    dfloat *__restrict weighted_counts,      //
    dfloat *__restrict sin2phi,              //
    dfloat *__restrict cos2phi,              //
    dfloat *__restrict weighted_sin_sq,      //
    dfloat *__restrict weighted_cos_sq,      //
    dfloat *__restrict weighted_sincos,      //
    dfloat *__restrict weighted_sin,         //
    dfloat *__restrict weighted_cos,         //
    dfloat *__restrict one_over_determinant, //
    const MPI_Comm comm                      //
) {

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < nsamples; ++idx) {
    dfloat angle = pol_angles[idx];
    sin2phi[idx] = std::sin(2.0 * angle);
    cos2phi[idx] = std::cos(2.0 * angle);
  } // for

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < nsamples; ++idx) {
    ssize_t pixel = pointings[idx];
    dfloat weight = pointings_flag[idx] * noise_weights[idx];

    dfloat wsin = weight * sin2phi[idx];
    dfloat wsin_sq = weight * sin2phi[idx] * sin2phi[idx];
    dfloat wcos = weight * cos2phi[idx];
    dfloat wcos_sq = weight * cos2phi[idx] * cos2phi[idx];
    dfloat wsincos = weight * sin2phi[idx] * cos2phi[idx];

#pragma omp atomic update
    hit_counts[pixel] += pointings_flag[idx];
#pragma omp atomic update
    weighted_counts[pixel] += weight;
#pragma omp atomic update
    weighted_sin[pixel] += wsin;
#pragma omp atomic update
    weighted_sin_sq[pixel] += wsin_sq;
#pragma omp atomic update
    weighted_cos[pixel] += wcos;
#pragma omp atomic update
    weighted_cos_sq[pixel] += wcos_sq;
#pragma omp atomic update
    weighted_sincos[pixel] += wsincos;

  } // for

  MPI_Allreduce(MPI_IN_PLACE, hit_counts, npix, mpi_get_type<dint>(), MPI_SUM,
                comm);
  MPI_Allreduce(MPI_IN_PLACE, weighted_counts, npix, mpi_get_type<dfloat>(),
                MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, weighted_sin, npix, mpi_get_type<dfloat>(),
                MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, weighted_cos, npix, mpi_get_type<dfloat>(),
                MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, weighted_sin_sq, npix, mpi_get_type<dfloat>(),
                MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, weighted_cos_sq, npix, mpi_get_type<dfloat>(),
                MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, weighted_sincos, npix, mpi_get_type<dfloat>(),
                MPI_SUM, comm);

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < npix; ++idx) {
    dfloat determinant =
        weighted_counts[idx] * weighted_cos_sq[idx] * weighted_sin_sq[idx] +
        2.0 * weighted_cos[idx] * weighted_sin[idx] * weighted_sincos[idx] -
        weighted_counts[idx] * weighted_sincos[idx] * weighted_sincos[idx] -
        weighted_cos[idx] * weighted_cos[idx] * weighted_sin_sq[idx] -
        weighted_sin[idx] * weighted_sin[idx] * weighted_cos_sq[idx];

    one_over_determinant[idx] = determinant;
  } // for

  return;

} // compute_weights_pol_IQU()

template <typename dint, typename dfloat>
dint get_pixel_mask_pol(                           //
    const int solver_type,                         //
    const ssize_t npix,                            //
    const dfloat threshold,                        //
    const dint *__restrict hit_counts,             //
    const dfloat *__restrict one_over_determinant, //
    dint *__restrict observed_pixels,              //
    dint *__restrict __old2new_pixel,              //
    bool *__restrict pixel_flag                    //
) {

  int hit_threshold = solver_type - 1;

  dint new_npix = 0;
  for (ssize_t idx = 0; idx < npix; ++idx) {

    dint hits = hit_counts[idx];
    dfloat determinant = one_over_determinant[idx];

    if ((abs(determinant) > threshold) && (hits > hit_threshold)) {
      observed_pixels[new_npix] = idx;
      __old2new_pixel[idx] = new_npix;
      pixel_flag[idx] = true;
      ++new_npix;
    } // if

  } // for

  return new_npix;

} // get_pixel_mask_pol()

template <template <typename, int = py::array::c_style> class buffer_t,
          typename dint,
          typename dfloat>
std::function<dint(                       //
    const ssize_t npix,                   //
    const ssize_t nsamples,               //
    const buffer_t<dint> pointings,       //
    const buffer_t<bool> pointings_flag,  //
    const buffer_t<dfloat> noise_weights, //
    buffer_t<dint> hit_counts,            //
    buffer_t<dfloat> weighted_counts,     //
    buffer_t<dint> observed_pixels,       //
    buffer_t<dint> __old2new_pixel,       //
    buffer_t<bool> pixel_flag,            //
    const py::object mpi4py_comm          //
    )>
    numpy_bind_compute_weights_pol_I =  //
    [](const ssize_t npix,              //
       const ssize_t nsamples,          //
       const py::buffer pointings,      //
       const py::buffer pointings_flag, //
       const py::buffer noise_weights,  //
       py::buffer hit_counts,           //
       py::buffer weighted_counts,      //
       py::buffer observed_pixels,      //
       py::buffer __old2new_pixel,      //
       py::buffer pixel_flag,           //
       const py::object mpi4py_comm     //
       ) -> dint {
  py::buffer_info pointings_info = pointings.request();
  py::buffer_info pointings_flags_info = pointings_flag.request();
  py::buffer_info noise_weights_info = noise_weights.request();
  py::buffer_info hit_counts_info = hit_counts.request();
  py::buffer_info weighted_counts_info = weighted_counts.request();
  py::buffer_info observed_pixels_info = observed_pixels.request();
  py::buffer_info __old2new_pixel_info = __old2new_pixel.request();
  py::buffer_info pixel_flag_info = pixel_flag.request();

  const dint *pointings_ptr =
      reinterpret_cast<const dint *>(pointings_info.ptr);
  const bool *pointings_flags_ptr =
      reinterpret_cast<const bool *>(pointings_flags_info.ptr);
  const dfloat *noise_weights_ptr =
      reinterpret_cast<const dfloat *>(noise_weights_info.ptr);
  dint *hit_counts_ptr = reinterpret_cast<dint *>(hit_counts_info.ptr);
  dfloat *weighted_counts_ptr =
      reinterpret_cast<dfloat *>(weighted_counts_info.ptr);
  dint *observed_pixels_ptr =
      reinterpret_cast<dint *>(observed_pixels_info.ptr);
  dint *__old2new_pixel_ptr =
      reinterpret_cast<dint *>(__old2new_pixel_info.ptr);
  bool *pixel_flag_ptr = reinterpret_cast<bool *>(pixel_flag_info.ptr);

  const MPI_Comm comm =
      (reinterpret_cast<const PyMPICommObject *>(mpi4py_comm.ptr()))->ob_mpi;

  dint new_npix = compute_weights_pol_I( //
      npix,                              //
      nsamples,                          //
      pointings_ptr,                     //
      pointings_flags_ptr,               //
      noise_weights_ptr,                 //
      hit_counts_ptr,                    //
      weighted_counts_ptr,               //
      observed_pixels_ptr,               //
      __old2new_pixel_ptr,               //
      pixel_flag_ptr,                    //
      comm                               //
  );

  return new_npix;
}; // numpy_bind_compute_weights_pol_I

template <template <typename, int = py::array::c_style> class buffer_t,
          typename dint, typename dfloat>
std::function<void(                        //
    const ssize_t npix,                    //
    const ssize_t nsamples,                //
    const buffer_t<dint> pointings,        //
    const buffer_t<bool> pointings_flag,   //
    const buffer_t<dfloat> noise_weights,  //
    const buffer_t<dfloat> pol_angles,     //
    buffer_t<dint> hit_counts,             //
    buffer_t<dfloat> weighted_counts,      //
    buffer_t<dfloat> sin2phi,              //
    buffer_t<dfloat> cos2phi,              //
    buffer_t<dfloat> weighted_sin_sq,      //
    buffer_t<dfloat> weighted_cos_sq,      //
    buffer_t<dfloat> weighted_sincos,      //
    buffer_t<dfloat> one_over_determinant, //
    const py::object mpi4py_comm           //
    )>
    numpy_bind_compute_weights_pol_QU = //
    [](const ssize_t npix,              //
       const ssize_t nsamples,          //
       const py::buffer pointings,      //
       const py::buffer pointings_flag, //
       const py::buffer noise_weights,  //
       const py::buffer pol_angles,     //
       py::buffer hit_counts,           //
       py::buffer weighted_counts,      //
       py::buffer sin2phi,              //
       py::buffer cos2phi,              //
       py::buffer weighted_sin_sq,      //
       py::buffer weighted_cos_sq,      //
       py::buffer weighted_sincos,      //
       py::buffer one_over_determinant, //
       const py::object mpi4py_comm     //
    ) {
      py::buffer_info pointings_info = pointings.request();
      py::buffer_info pointings_flag_info = pointings_flag.request();
      py::buffer_info noise_weights_info = noise_weights.request();
      py::buffer_info pol_angles_info = pol_angles.request();
      py::buffer_info hit_counts_info = hit_counts.request();
      py::buffer_info weighted_counts_info = weighted_counts.request();
      py::buffer_info sin2phi_info = sin2phi.request();
      py::buffer_info cos2phi_info = cos2phi.request();
      py::buffer_info weighted_sin_sq_info = weighted_sin_sq.request();
      py::buffer_info weighted_cos_sq_info = weighted_cos_sq.request();
      py::buffer_info weighted_sincos_info = weighted_sincos.request();
      py::buffer_info one_over_determinant_info =
          one_over_determinant.request();

      const dint *pointings_ptr =
          reinterpret_cast<const dint *>(pointings_info.ptr);
      const bool *pointings_flag_ptr =
          reinterpret_cast<const bool *>(pointings_flag_info.ptr);
      const dfloat *noise_weights_ptr =
          reinterpret_cast<const dfloat *>(noise_weights_info.ptr);
      const dfloat *pol_angles_ptr =
          reinterpret_cast<const dfloat *>(pol_angles_info.ptr);
      dint *hit_counts_ptr = reinterpret_cast<dint *>(hit_counts_info.ptr);
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
      dfloat *one_over_determinant_ptr =
          reinterpret_cast<dfloat *>(one_over_determinant_info.ptr);

      const MPI_Comm comm =
          (reinterpret_cast<const PyMPICommObject *>(mpi4py_comm.ptr()))
              ->ob_mpi;

      compute_weights_pol_QU(       //
          npix,                     //
          nsamples,                 //
          pointings_ptr,            //
          pointings_flag_ptr,       //
          noise_weights_ptr,        //
          pol_angles_ptr,           //
          hit_counts_ptr,           //
          weighted_counts_ptr,      //
          sin2phi_ptr,              //
          cos2phi_ptr,              //
          weighted_sin_sq_ptr,      //
          weighted_cos_sq_ptr,      //
          weighted_sincos_ptr,      //
          one_over_determinant_ptr, //
          comm                      //
      );

      return;
    }; // numpy_bind_compute_weights_pol_QU()

template <template <typename, int = py::array::c_style> class buffer_t,
          typename dint,
          typename dfloat>
std::function<void(                        //
    const ssize_t npix,                    //
    const ssize_t nsamples,                //
    const buffer_t<dint> pointings,        //
    const buffer_t<bool> pointings_flag,   //
    const buffer_t<dfloat> noise_weights,  //
    const buffer_t<dfloat> pol_angles,     //
    buffer_t<dint> hit_counts,             //
    buffer_t<dfloat> weighted_counts,      //
    buffer_t<dfloat> sin2phi,              //
    buffer_t<dfloat> cos2phi,              //
    buffer_t<dfloat> weighted_sin_sq,      //
    buffer_t<dfloat> weighted_cos_sq,      //
    buffer_t<dfloat> weighted_sincos,      //
    buffer_t<dfloat> weighted_sin,         //
    buffer_t<dfloat> weighted_cos,         //
    buffer_t<dfloat> one_over_determinant, //
    const py::object mpi4py_comm           //
    )>
    numpy_bind_compute_weights_pol_IQU = //
    [](const ssize_t npix,               //
       const ssize_t nsamples,           //
       const py::buffer pointings,       //
       const py::buffer pointings_flag,  //
       const py::buffer noise_weights,   //
       const py::buffer pol_angles,      //
       py::buffer hit_counts,            //
       py::buffer weighted_counts,       //
       py::buffer sin2phi,               //
       py::buffer cos2phi,               //
       py::buffer weighted_sin_sq,       //
       py::buffer weighted_cos_sq,       //
       py::buffer weighted_sincos,       //
       py::buffer weighted_sin,          //
       py::buffer weighted_cos,          //
       py::buffer one_over_determinant,  //
       const py::object mpi4py_comm      //
    ) {
      py::buffer_info pointings_info = pointings.request();
      py::buffer_info pointings_flag_info = pointings_flag.request();
      py::buffer_info noise_weights_info = noise_weights.request();
      py::buffer_info pol_angles_info = pol_angles.request();
      py::buffer_info hit_counts_info = hit_counts.request();
      py::buffer_info weighted_counts_info = weighted_counts.request();
      py::buffer_info sin2phi_info = sin2phi.request();
      py::buffer_info cos2phi_info = cos2phi.request();
      py::buffer_info weighted_sin_sq_info = weighted_sin_sq.request();
      py::buffer_info weighted_cos_sq_info = weighted_cos_sq.request();
      py::buffer_info weighted_sincos_info = weighted_sincos.request();
      py::buffer_info weighted_sin_info = weighted_sin.request();
      py::buffer_info weighted_cos_info = weighted_cos.request();
      py::buffer_info one_over_determinant_info =
          one_over_determinant.request();

      const dint *pointings_ptr =
          reinterpret_cast<const dint *>(pointings_info.ptr);
      const bool *pointings_flag_ptr =
          reinterpret_cast<const bool *>(pointings_flag_info.ptr);
      const dfloat *noise_weights_ptr =
          reinterpret_cast<const dfloat *>(noise_weights_info.ptr);
      const dfloat *pol_angles_ptr =
          reinterpret_cast<const dfloat *>(pol_angles_info.ptr);
      dint *hit_counts_ptr = reinterpret_cast<dint *>(hit_counts_info.ptr);
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
      dfloat *one_over_determinant_ptr =
          reinterpret_cast<dfloat *>(one_over_determinant_info.ptr);

      const MPI_Comm comm =
          (reinterpret_cast<const PyMPICommObject *>(mpi4py_comm.ptr()))
              ->ob_mpi;

      compute_weights_pol_IQU(      //
          npix,                     //
          nsamples,                 //
          pointings_ptr,            //
          pointings_flag_ptr,       //
          noise_weights_ptr,        //
          pol_angles_ptr,           //
          hit_counts_ptr,           //
          weighted_counts_ptr,      //
          sin2phi_ptr,              //
          cos2phi_ptr,              //
          weighted_sin_sq_ptr,      //
          weighted_cos_sq_ptr,      //
          weighted_sincos_ptr,      //
          weighted_sin_ptr,         //
          weighted_cos_ptr,         //
          one_over_determinant_ptr, //
          comm                      //
      );

      return;
    }; // numpy_bind_compute_weights_pol_IQU()

template <template <typename, int = py::array::c_style> class buffer_t,
          typename dint,
          typename dfloat>
std::function<dint(                              //
    const int solver_type,                       //
    const ssize_t npix,                          //
    const dfloat threshold,                      //
    const buffer_t<dint> hit_counts,             //
    const buffer_t<dfloat> one_over_determinant, //
    buffer_t<dint> observed_pixels,              //
    buffer_t<dint> __old2new_pixel,              //
    buffer_t<bool> pixel_flag                    //
    )>
    numpy_bind_get_pixel_mask_pol =           //
    [](const int solver_type,                 //
       const ssize_t npix,                    //
       const dfloat threshold,                //
       const py::buffer hit_counts,           //
       const py::buffer one_over_determinant, //
       py::buffer observed_pixels,            //
       py::buffer __old2new_pixel,            //
       py::buffer pixel_flag                  //
       ) -> dint {
  py::buffer_info hit_counts_info = hit_counts.request();
  py::buffer_info one_over_determinant_info = one_over_determinant.request();
  py::buffer_info observed_pixels_info = observed_pixels.request();
  py::buffer_info __old2new_pixel_info = __old2new_pixel.request();
  py::buffer_info pixel_flag_info = pixel_flag.request();

  const dint *hit_counts_ptr =
      reinterpret_cast<const dint *>(hit_counts_info.ptr);
  const dfloat *one_over_determinant_ptr =
      reinterpret_cast<const dfloat *>(one_over_determinant_info.ptr);
  dint *observed_pixels_ptr =
      reinterpret_cast<dint *>(observed_pixels_info.ptr);
  dint *__old2new_pixel_ptr =
      reinterpret_cast<dint *>(__old2new_pixel_info.ptr);
  bool *pixel_flag_ptr = reinterpret_cast<bool *>(pixel_flag_info.ptr);

  dint new_npix = get_pixel_mask_pol( //
      solver_type,                    //
      npix,                           //
      threshold,                      //
      hit_counts_ptr,                 //
      one_over_determinant_ptr,       //
      observed_pixels_ptr,            //
      __old2new_pixel_ptr,            //
      pixel_flag_ptr                  //
  );

  return new_npix;
}; // numpy_bind_get_pixel_mask_pol()

PYBIND11_MODULE(compute_weights, m) {
  m.doc() = "compute_weights";
  m.def("compute_weights_pol_I",
        numpy_bind_compute_weights_pol_I<py::array_t, int32_t, float>,
        py::arg("npix"),                        //
        py::arg("nsamples"),                    //
        py::arg("pointings").noconvert(),       //
        py::arg("pointings_flag").noconvert(),  //
        py::arg("noise_weights").noconvert(),   //
        py::arg("hit_counts").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("observed_pixels").noconvert(), //
        py::arg("__old2new_pixel").noconvert(), //
        py::arg("pixel_flag").noconvert(),      //
        py::arg("comm").noconvert()             //
  );
  m.def("compute_weights_pol_I",
        numpy_bind_compute_weights_pol_I<py::array_t, int64_t, float>,
        py::arg("npix"),                        //
        py::arg("nsamples"),                    //
        py::arg("pointings").noconvert(),       //
        py::arg("pointings_flag").noconvert(),  //
        py::arg("noise_weights").noconvert(),   //
        py::arg("hit_counts").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("observed_pixels").noconvert(), //
        py::arg("__old2new_pixel").noconvert(), //
        py::arg("pixel_flag").noconvert(),      //
        py::arg("comm").noconvert()             //
  );
  m.def("compute_weights_pol_I",
        numpy_bind_compute_weights_pol_I<py::array_t, int32_t, double>,
        py::arg("npix"),                        //
        py::arg("nsamples"),                    //
        py::arg("pointings").noconvert(),       //
        py::arg("pointings_flag").noconvert(),  //
        py::arg("noise_weights").noconvert(),   //
        py::arg("hit_counts").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("observed_pixels").noconvert(), //
        py::arg("__old2new_pixel").noconvert(), //
        py::arg("pixel_flag").noconvert(),      //
        py::arg("comm").noconvert()             //
  );
  m.def("compute_weights_pol_I",
        numpy_bind_compute_weights_pol_I<py::array_t, int64_t, double>,
        py::arg("npix"),                        //
        py::arg("nsamples"),                    //
        py::arg("pointings").noconvert(),       //
        py::arg("pointings_flag").noconvert(),  //
        py::arg("noise_weights").noconvert(),   //
        py::arg("hit_counts").noconvert(),      //
        py::arg("weighted_counts").noconvert(), //
        py::arg("observed_pixels").noconvert(), //
        py::arg("__old2new_pixel").noconvert(), //
        py::arg("pixel_flag").noconvert(),      //
        py::arg("comm").noconvert()             //
  );

  m.def("compute_weights_pol_QU",
        numpy_bind_compute_weights_pol_QU<py::array_t, int32_t, float>,
        py::arg("npix"),                             //
        py::arg("nsamples"),                         //
        py::arg("pointings").noconvert(),            //
        py::arg("pointings_flag").noconvert(),       //
        py::arg("noise_weights").noconvert(),        //
        py::arg("pol_angles").noconvert(),           //
        py::arg("hit_counts").noconvert(),           //
        py::arg("weighted_counts").noconvert(),      //
        py::arg("sin2phi").noconvert(),              //
        py::arg("cos2phi").noconvert(),              //
        py::arg("weighted_sin_sq").noconvert(),      //
        py::arg("weighted_cos_sq").noconvert(),      //
        py::arg("weighted_sincos").noconvert(),      //
        py::arg("one_over_determinant").noconvert(), //
        py::arg("comm").noconvert()                  //
  );

  m.def("compute_weights_pol_QU",
        numpy_bind_compute_weights_pol_QU<py::array_t, int64_t, float>,
        py::arg("npix"),                             //
        py::arg("nsamples"),                         //
        py::arg("pointings").noconvert(),            //
        py::arg("pointings_flag").noconvert(),       //
        py::arg("noise_weights").noconvert(),        //
        py::arg("pol_angles").noconvert(),           //
        py::arg("hit_counts").noconvert(),           //
        py::arg("weighted_counts").noconvert(),      //
        py::arg("sin2phi").noconvert(),              //
        py::arg("cos2phi").noconvert(),              //
        py::arg("weighted_sin_sq").noconvert(),      //
        py::arg("weighted_cos_sq").noconvert(),      //
        py::arg("weighted_sincos").noconvert(),      //
        py::arg("one_over_determinant").noconvert(), //
        py::arg("comm").noconvert()                  //
  );

  m.def("compute_weights_pol_QU",
        numpy_bind_compute_weights_pol_QU<py::array_t, int32_t, double>,
        py::arg("npix"),                             //
        py::arg("nsamples"),                         //
        py::arg("pointings").noconvert(),            //
        py::arg("pointings_flag").noconvert(),       //
        py::arg("noise_weights").noconvert(),        //
        py::arg("pol_angles").noconvert(),           //
        py::arg("hit_counts").noconvert(),           //
        py::arg("weighted_counts").noconvert(),      //
        py::arg("sin2phi").noconvert(),              //
        py::arg("cos2phi").noconvert(),              //
        py::arg("weighted_sin_sq").noconvert(),      //
        py::arg("weighted_cos_sq").noconvert(),      //
        py::arg("weighted_sincos").noconvert(),      //
        py::arg("one_over_determinant").noconvert(), //
        py::arg("comm").noconvert()                  //
  );

  m.def("compute_weights_pol_QU",
        numpy_bind_compute_weights_pol_QU<py::array_t, int64_t, double>,
        py::arg("npix"),                             //
        py::arg("nsamples"),                         //
        py::arg("pointings").noconvert(),            //
        py::arg("pointings_flag").noconvert(),       //
        py::arg("noise_weights").noconvert(),        //
        py::arg("pol_angles").noconvert(),           //
        py::arg("hit_counts").noconvert(),           //
        py::arg("weighted_counts").noconvert(),      //
        py::arg("sin2phi").noconvert(),              //
        py::arg("cos2phi").noconvert(),              //
        py::arg("weighted_sin_sq").noconvert(),      //
        py::arg("weighted_cos_sq").noconvert(),      //
        py::arg("weighted_sincos").noconvert(),      //
        py::arg("one_over_determinant").noconvert(), //
        py::arg("comm").noconvert()                  //
  );

  m.def("compute_weights_pol_IQU",
        numpy_bind_compute_weights_pol_IQU<py::array_t, int32_t, float>,
        py::arg("npix"),                             //
        py::arg("nsamples"),                         //
        py::arg("pointings").noconvert(),            //
        py::arg("pointings_flag").noconvert(),       //
        py::arg("noise_weights").noconvert(),        //
        py::arg("pol_angles").noconvert(),           //
        py::arg("hit_counts").noconvert(),           //
        py::arg("weighted_counts").noconvert(),      //
        py::arg("sin2phi").noconvert(),              //
        py::arg("cos2phi").noconvert(),              //
        py::arg("weighted_sin_sq").noconvert(),      //
        py::arg("weighted_cos_sq").noconvert(),      //
        py::arg("weighted_sincos").noconvert(),      //
        py::arg("weighted_sin").noconvert(),         //
        py::arg("weighted_cos").noconvert(),         //
        py::arg("one_over_determinant").noconvert(), //
        py::arg("comm").noconvert()                  //
  );

  m.def("compute_weights_pol_IQU",
        numpy_bind_compute_weights_pol_IQU<py::array_t, int64_t, float>,
        py::arg("npix"),                             //
        py::arg("nsamples"),                         //
        py::arg("pointings").noconvert(),            //
        py::arg("pointings_flag").noconvert(),       //
        py::arg("noise_weights").noconvert(),        //
        py::arg("pol_angles").noconvert(),           //
        py::arg("hit_counts").noconvert(),           //
        py::arg("weighted_counts").noconvert(),      //
        py::arg("sin2phi").noconvert(),              //
        py::arg("cos2phi").noconvert(),              //
        py::arg("weighted_sin_sq").noconvert(),      //
        py::arg("weighted_cos_sq").noconvert(),      //
        py::arg("weighted_sincos").noconvert(),      //
        py::arg("weighted_sin").noconvert(),         //
        py::arg("weighted_cos").noconvert(),         //
        py::arg("one_over_determinant").noconvert(), //
        py::arg("comm").noconvert()                  //
  );

  m.def("compute_weights_pol_IQU",
        numpy_bind_compute_weights_pol_IQU<py::array_t, int32_t, double>,
        py::arg("npix"),                             //
        py::arg("nsamples"),                         //
        py::arg("pointings").noconvert(),            //
        py::arg("pointings_flag").noconvert(),       //
        py::arg("noise_weights").noconvert(),        //
        py::arg("pol_angles").noconvert(),           //
        py::arg("hit_counts").noconvert(),           //
        py::arg("weighted_counts").noconvert(),      //
        py::arg("sin2phi").noconvert(),              //
        py::arg("cos2phi").noconvert(),              //
        py::arg("weighted_sin_sq").noconvert(),      //
        py::arg("weighted_cos_sq").noconvert(),      //
        py::arg("weighted_sincos").noconvert(),      //
        py::arg("weighted_sin").noconvert(),         //
        py::arg("weighted_cos").noconvert(),         //
        py::arg("one_over_determinant").noconvert(), //
        py::arg("comm").noconvert()                  //
  );

  m.def("compute_weights_pol_IQU",
        numpy_bind_compute_weights_pol_IQU<py::array_t, int64_t, double>,
        py::arg("npix"),                             //
        py::arg("nsamples"),                         //
        py::arg("pointings").noconvert(),            //
        py::arg("pointings_flag").noconvert(),       //
        py::arg("noise_weights").noconvert(),        //
        py::arg("pol_angles").noconvert(),           //
        py::arg("hit_counts").noconvert(),           //
        py::arg("weighted_counts").noconvert(),      //
        py::arg("sin2phi").noconvert(),              //
        py::arg("cos2phi").noconvert(),              //
        py::arg("weighted_sin_sq").noconvert(),      //
        py::arg("weighted_cos_sq").noconvert(),      //
        py::arg("weighted_sincos").noconvert(),      //
        py::arg("weighted_sin").noconvert(),         //
        py::arg("weighted_cos").noconvert(),         //
        py::arg("one_over_determinant").noconvert(), //
        py::arg("comm").noconvert()                  //
  );

  m.def("get_pixel_mask_pol",
        numpy_bind_get_pixel_mask_pol<py::array_t, int32_t, float>,
        py::arg("solver_type"),                      //
        py::arg("npix"),                             //
        py::arg("threshold"),                        //
        py::arg("hit_counts").noconvert(),           //
        py::arg("one_over_determinant").noconvert(), //
        py::arg("observed_pixels").noconvert(),      //
        py::arg("__old2new_pixel").noconvert(),      //
        py::arg("pixel_flag").noconvert()            //
  );
  m.def("get_pixel_mask_pol",
        numpy_bind_get_pixel_mask_pol<py::array_t, int64_t, float>,
        py::arg("solver_type"),                      //
        py::arg("npix"),                             //
        py::arg("threshold"),                        //
        py::arg("hit_counts").noconvert(),           //
        py::arg("one_over_determinant").noconvert(), //
        py::arg("observed_pixels").noconvert(),      //
        py::arg("__old2new_pixel").noconvert(),      //
        py::arg("pixel_flag").noconvert()            //
  );
  m.def("get_pixel_mask_pol",
        numpy_bind_get_pixel_mask_pol<py::array_t, int32_t, double>,
        py::arg("solver_type"),                      //
        py::arg("npix"),                             //
        py::arg("threshold"),                        //
        py::arg("hit_counts").noconvert(),           //
        py::arg("one_over_determinant").noconvert(), //
        py::arg("observed_pixels").noconvert(),      //
        py::arg("__old2new_pixel").noconvert(),      //
        py::arg("pixel_flag").noconvert()            //
  );
  m.def("get_pixel_mask_pol",
        numpy_bind_get_pixel_mask_pol<py::array_t, int64_t, double>,
        py::arg("solver_type"),                      //
        py::arg("npix"),                             //
        py::arg("threshold"),                        //
        py::arg("hit_counts").noconvert(),           //
        py::arg("one_over_determinant").noconvert(), //
        py::arg("observed_pixels").noconvert(),      //
        py::arg("__old2new_pixel").noconvert(),      //
        py::arg("pixel_flag").noconvert()            //
  );
}
