#include <cmath>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#ifndef _DISABLE_OMP
#include <omp.h>
#endif

#include "mpi_utils.hpp"

namespace nb = nanobind;

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
  } // for

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

template <typename dint, typename dfloat, typename device> //
void register_compute_weights(nb::module_ &m) {
  using arr_dint = nb::ndarray<dint, nb::ndim<1>, device, nb::c_contig>;
  using arr_dfloat = nb::ndarray<dfloat, nb::ndim<1>, device, nb::c_contig>;
  using arr_bool = nb::ndarray<bool, nb::ndim<1>, device, nb::c_contig>;

  auto get_comm = [](const nb::object &mpi4py_comm) -> MPI_Comm {
    return (reinterpret_cast<const PyMPICommObject *>(mpi4py_comm.ptr()))
        ->ob_mpi;
  };

  m.def(
      "compute_weights_pol_I",            //
      [get_comm](                         //
          const ssize_t npix,             //
          const ssize_t nsamples,         //
          const arr_dint pointings,       //
          const arr_bool pointings_flag,  //
          const arr_dfloat noise_weights, //
          arr_dint hit_counts,            //
          arr_dfloat weighted_counts,     //
          arr_dint observed_pixels,       //
          arr_dint __old2new_pixel,       //
          arr_bool pixel_flag,            //
          const nb::object mpi4py_comm    //
          ) -> dint {
        return compute_weights_pol_I( //
            npix,                     //
            nsamples,                 //
            pointings.data(),         //
            pointings_flag.data(),    //
            noise_weights.data(),     //
            hit_counts.data(),        //
            weighted_counts.data(),   //
            observed_pixels.data(),   //
            __old2new_pixel.data(),   //
            pixel_flag.data(),        //
            get_comm(mpi4py_comm)     //
        );
      },
      nb::arg("npix"),                        //
      nb::arg("nsamples"),                    //
      nb::arg("pointings").noconvert(),       //
      nb::arg("pointings_flag").noconvert(),  //
      nb::arg("noise_weights").noconvert(),   //
      nb::arg("hit_counts").noconvert(),      //
      nb::arg("weighted_counts").noconvert(), //
      nb::arg("observed_pixels").noconvert(), //
      nb::arg("__old2new_pixel").noconvert(), //
      nb::arg("pixel_flag").noconvert(),      //
      nb::arg("comm").noconvert()             //
  );

  m.def(
      "compute_weights_pol_QU",            //
      [get_comm](                          //
          const ssize_t npix,              //
          const ssize_t nsamples,          //
          const arr_dint pointings,        //
          const arr_bool pointings_flag,   //
          const arr_dfloat noise_weights,  //
          const arr_dfloat pol_angles,     //
          arr_dint hit_counts,             //
          arr_dfloat weighted_counts,      //
          arr_dfloat sin2phi,              //
          arr_dfloat cos2phi,              //
          arr_dfloat weighted_sin_sq,      //
          arr_dfloat weighted_cos_sq,      //
          arr_dfloat weighted_sincos,      //
          arr_dfloat one_over_determinant, //
          const nb::object mpi4py_comm     //
      ) {
        compute_weights_pol_QU(          //
            npix,                        //
            nsamples,                    //
            pointings.data(),            //
            pointings_flag.data(),       //
            noise_weights.data(),        //
            pol_angles.data(),           //
            hit_counts.data(),           //
            weighted_counts.data(),      //
            sin2phi.data(),              //
            cos2phi.data(),              //
            weighted_sin_sq.data(),      //
            weighted_cos_sq.data(),      //
            weighted_sincos.data(),      //
            one_over_determinant.data(), //
            get_comm(mpi4py_comm)        //
        );
      },
      nb::arg("npix"),                             //
      nb::arg("nsamples"),                         //
      nb::arg("pointings").noconvert(),            //
      nb::arg("pointings_flag").noconvert(),       //
      nb::arg("noise_weights").noconvert(),        //
      nb::arg("pol_angles").noconvert(),           //
      nb::arg("hit_counts").noconvert(),           //
      nb::arg("weighted_counts").noconvert(),      //
      nb::arg("sin2phi").noconvert(),              //
      nb::arg("cos2phi").noconvert(),              //
      nb::arg("weighted_sin_sq").noconvert(),      //
      nb::arg("weighted_cos_sq").noconvert(),      //
      nb::arg("weighted_sincos").noconvert(),      //
      nb::arg("one_over_determinant").noconvert(), //
      nb::arg("comm").noconvert()                  //
  );

  m.def(
      "compute_weights_pol_IQU",           //
      [get_comm](                          //
          const ssize_t npix,              //
          const ssize_t nsamples,          //
          const arr_dint pointings,        //
          const arr_bool pointings_flag,   //
          const arr_dfloat noise_weights,  //
          const arr_dfloat pol_angles,     //
          arr_dint hit_counts,             //
          arr_dfloat weighted_counts,      //
          arr_dfloat sin2phi,              //
          arr_dfloat cos2phi,              //
          arr_dfloat weighted_sin_sq,      //
          arr_dfloat weighted_cos_sq,      //
          arr_dfloat weighted_sincos,      //
          arr_dfloat weighted_sin,         //
          arr_dfloat weighted_cos,         //
          arr_dfloat one_over_determinant, //
          const nb::object mpi4py_comm     //
      ) {
        compute_weights_pol_IQU(         //
            npix,                        //
            nsamples,                    //
            pointings.data(),            //
            pointings_flag.data(),       //
            noise_weights.data(),        //
            pol_angles.data(),           //
            hit_counts.data(),           //
            weighted_counts.data(),      //
            sin2phi.data(),              //
            cos2phi.data(),              //
            weighted_sin_sq.data(),      //
            weighted_cos_sq.data(),      //
            weighted_sincos.data(),      //
            weighted_sin.data(),         //
            weighted_cos.data(),         //
            one_over_determinant.data(), //
            get_comm(mpi4py_comm)        //
        );
      },
      nb::arg("npix"),                             //
      nb::arg("nsamples"),                         //
      nb::arg("pointings").noconvert(),            //
      nb::arg("pointings_flag").noconvert(),       //
      nb::arg("noise_weights").noconvert(),        //
      nb::arg("pol_angles").noconvert(),           //
      nb::arg("hit_counts").noconvert(),           //
      nb::arg("weighted_counts").noconvert(),      //
      nb::arg("sin2phi").noconvert(),              //
      nb::arg("cos2phi").noconvert(),              //
      nb::arg("weighted_sin_sq").noconvert(),      //
      nb::arg("weighted_cos_sq").noconvert(),      //
      nb::arg("weighted_sincos").noconvert(),      //
      nb::arg("weighted_sin").noconvert(),         //
      nb::arg("weighted_cos").noconvert(),         //
      nb::arg("one_over_determinant").noconvert(), //
      nb::arg("comm").noconvert()                  //
  );

  m.def(
      "get_pixel_mask_pol",                      //
      [](                                        //
          const int solver_type,                 //
          const ssize_t npix,                    //
          const dfloat threshold,                //
          const arr_dint hit_counts,             //
          const arr_dfloat one_over_determinant, //
          arr_dint observed_pixels,              //
          arr_dint __old2new_pixel,              //
          arr_bool pixel_flag                    //
          ) -> dint {
        return get_pixel_mask_pol(       //
            solver_type,                 //
            npix,                        //
            threshold,                   //
            hit_counts.data(),           //
            one_over_determinant.data(), //
            observed_pixels.data(),      //
            __old2new_pixel.data(),      //
            pixel_flag.data()            //
        );
      },
      nb::arg("solver_type"),                      //
      nb::arg("npix"),                             //
      nb::arg("threshold"),                        //
      nb::arg("hit_counts").noconvert(),           //
      nb::arg("one_over_determinant").noconvert(), //
      nb::arg("observed_pixels").noconvert(),      //
      nb::arg("__old2new_pixel").noconvert(),      //
      nb::arg("pixel_flag").noconvert()            //
  );
}

NB_MODULE(compute_weights, m) {
  m.doc() = "compute_weights";

  register_compute_weights<int64_t, double, nb::device::cpu>(m);
  register_compute_weights<int32_t, double, nb::device::cpu>(m);
  register_compute_weights<int64_t, float, nb::device::cpu>(m);
  register_compute_weights<int32_t, float, nb::device::cpu>(m);
}
