#include <cmath>
#include <cstddef>

//////////////////////////////////
// Local accumulation functions //
//////////////////////////////////

template <typename dint, typename dfloat>
void accumulate_weights_pol_I(              //
    const ssize_t nsamples,                 //
    const dint *__restrict pointings,       //
    const bool *__restrict pointings_flag,  //
    const dfloat *__restrict noise_weights, //
    dint *__restrict hit_counts,            //
    dfloat *__restrict weighted_counts      //
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

  return;

} // accumulate_weights_pol_I()

template <typename dint, typename dfloat>
void accumulate_weights_pol_QU(             //
    const ssize_t nsamples,                 //
    const dint *__restrict pointings,       //
    const bool *__restrict pointings_flag,  //
    const dfloat *__restrict noise_weights, //
    const dfloat *__restrict pol_angles,    //
    const dfloat *__restrict sin2phi,       //
    const dfloat *__restrict cos2phi,       //
    dint *__restrict hit_counts,            //
    dfloat *__restrict weighted_counts,     //
    dfloat *__restrict weighted_sin_sq,     //
    dfloat *__restrict weighted_cos_sq,     //
    dfloat *__restrict weighted_sincos      //
) {

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

  return;

} // accumulate_weights_pol_QU()

template <typename dint, typename dfloat>
void accumulate_weights_pol_IQU(            //
    const ssize_t nsamples,                 //
    const dint *__restrict pointings,       //
    const bool *__restrict pointings_flag,  //
    const dfloat *__restrict noise_weights, //
    const dfloat *__restrict pol_angles,    //
    const dfloat *__restrict sin2phi,       //
    const dfloat *__restrict cos2phi,       //
    dint *__restrict hit_counts,            //
    dfloat *__restrict weighted_counts,     //
    dfloat *__restrict weighted_sin_sq,     //
    dfloat *__restrict weighted_cos_sq,     //
    dfloat *__restrict weighted_sincos,     //
    dfloat *__restrict weighted_sin,        //
    dfloat *__restrict weighted_cos         //
) {

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

  return;

} // accumulate_weights_pol_IQU()

///////////////////////////////////
// Compute determinant functions //
///////////////////////////////////

template <typename dint, typename dfloat>
void compute_determinants_pol_QU(           //
    const ssize_t npix,                     //
    dfloat *__restrict weighted_sin_sq,     //
    dfloat *__restrict weighted_cos_sq,     //
    dfloat *__restrict weighted_sincos,     //
    dfloat *__restrict one_over_determinant //
) {

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < npix; ++idx) {
    dfloat determinant = weighted_sin_sq[idx] * weighted_cos_sq[idx] -
                         weighted_sincos[idx] * weighted_sincos[idx];

    one_over_determinant[idx] = determinant;
  } // for

  return;

} // compute_determinants_pol_QU()

template <typename dint, typename dfloat>
void compute_determinants_pol_IQU(          //
    const ssize_t npix,                     //
    dfloat *__restrict weighted_counts,     //
    dfloat *__restrict weighted_sin_sq,     //
    dfloat *__restrict weighted_cos_sq,     //
    dfloat *__restrict weighted_sincos,     //
    dfloat *__restrict weighted_sin,        //
    dfloat *__restrict weighted_cos,        //
    dfloat *__restrict one_over_determinant //
) {

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

} // compute_determinants_pol_IQU()

////////////////////////////////////
// Node level pixel mask function //
////////////////////////////////////

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
