#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#ifndef _DISABLE_OMP
#include <omp.h>
#endif

namespace nb = nanobind;

template <typename dfloat>
void BDPLO_mult_QU(                                //
    const ssize_t new_npix,                        //
    const dfloat *__restrict weighted_sin_sq,      //
    const dfloat *__restrict weighted_cos_sq,      //
    const dfloat *__restrict weighted_sincos,      //
    const dfloat *__restrict one_over_determinant, //
    const dfloat *__restrict vec,                  //
    dfloat *__restrict prod                        //
) {

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < new_npix; ++idx) {
    prod[2 * idx] =
        one_over_determinant[idx] * (weighted_sin_sq[idx] * vec[2 * idx] -
                                     weighted_sincos[idx] * vec[2 * idx + 1]);
    prod[2 * idx + 1] =
        one_over_determinant[idx] * (weighted_cos_sq[idx] * vec[2 * idx + 1] -
                                     weighted_sincos[idx] * vec[2 * idx]);
  } // for

  return;
} // BDPLO_mult_QU()

template <typename dfloat>
void BDPLO_mult_IQU(                               //
    const ssize_t new_npix,                        //
    const dfloat *__restrict weighted_counts,      //
    const dfloat *__restrict weighted_sin_sq,      //
    const dfloat *__restrict weighted_cos_sq,      //
    const dfloat *__restrict weighted_sincos,      //
    const dfloat *__restrict weighted_sin,         //
    const dfloat *__restrict weighted_cos,         //
    const dfloat *__restrict one_over_determinant, //
    const dfloat *__restrict vec,                  //
    dfloat *__restrict prod                        //
) {

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < new_npix; ++idx) {

    prod[3 * idx] = one_over_determinant[idx] *
                    ((weighted_cos_sq[idx] * weighted_sin_sq[idx] -
                      weighted_sincos[idx] * weighted_sincos[idx]) *
                         vec[3 * idx] +
                     (weighted_sin[idx] * weighted_sincos[idx] -
                      weighted_cos[idx] * weighted_sin_sq[idx]) *
                         vec[3 * idx + 1] +
                     (weighted_cos[idx] * weighted_sincos[idx] -
                      weighted_sin[idx] * weighted_cos_sq[idx]) *
                         vec[3 * idx + 2]);
    prod[3 * idx + 1] = one_over_determinant[idx] *
                        ((weighted_sin[idx] * weighted_sincos[idx] -
                          weighted_cos[idx] * weighted_sin_sq[idx]) *
                             vec[3 * idx] +
                         (weighted_counts[idx] * weighted_sin_sq[idx] -
                          weighted_sin[idx] * weighted_sin[idx]) *
                             vec[3 * idx + 1] +
                         (weighted_sin[idx] * weighted_cos[idx] -
                          weighted_counts[idx] * weighted_sincos[idx]) *
                             vec[3 * idx + 2]);
    prod[3 * idx + 2] = one_over_determinant[idx] *
                        ((weighted_cos[idx] * weighted_sincos[idx] -
                          weighted_sin[idx] * weighted_cos_sq[idx]) *
                             vec[3 * idx] +
                         (weighted_cos[idx] * weighted_sin[idx] -
                          weighted_counts[idx] * weighted_sincos[idx]) *
                             vec[3 * idx + 1] +
                         (weighted_counts[idx] * weighted_cos_sq[idx] -
                          weighted_cos[idx] * weighted_cos[idx]) *
                             vec[3 * idx + 2]);
  } // for

  return;
} // BDPLO_mult_IQU()

template <typename dfloat, typename device> //
void register_BlkDiagPrecondLO(nb::module_ &m) {
  using arr_t = nb::ndarray<dfloat, nb::ndim<1>, device, nb::c_contig>;

  m.def(
      "BDPLO_mult_QU",                      //
      [](                                   //
          const ssize_t new_npix,           //
          const arr_t weighted_sin_sq,      //
          const arr_t weighted_cos_sq,      //
          const arr_t weighted_sincos,      //
          const arr_t one_over_determinant, //
          const arr_t vec,                  //
          arr_t prod                        //
      ) {
        BDPLO_mult_QU(                   //
            new_npix,                    //
            weighted_sin_sq.data(),      //
            weighted_cos_sq.data(),      //
            weighted_sincos.data(),      //
            one_over_determinant.data(), //
            vec.data(),                  //
            prod.data()                  //
        );
      },
      nb::arg("new_npix"),                         //
      nb::arg("weighted_sin_sq").noconvert(),      //
      nb::arg("weighted_cos_sq").noconvert(),      //
      nb::arg("weighted_sincos").noconvert(),      //
      nb::arg("one_over_determinant").noconvert(), //
      nb::arg("vec").noconvert(),                  //
      nb::arg("prod").noconvert()                  //
  );

  m.def(
      "BDPLO_mult_IQU",                     //
      [](                                   //
          const ssize_t new_npix,           //
          const arr_t weighted_counts,      //
          const arr_t weighted_sin_sq,      //
          const arr_t weighted_cos_sq,      //
          const arr_t weighted_sincos,      //
          const arr_t weighted_sin,         //
          const arr_t weighted_cos,         //
          const arr_t one_over_determinant, //
          const arr_t vec,                  //
          arr_t prod                        //
      ) {
        BDPLO_mult_IQU(                  //
            new_npix,                    //
            weighted_counts.data(),      //
            weighted_sin_sq.data(),      //
            weighted_cos_sq.data(),      //
            weighted_sincos.data(),      //
            weighted_sin.data(),         //
            weighted_cos.data(),         //
            one_over_determinant.data(), //
            vec.data(),                  //
            prod.data()                  //
        );
      },
      nb::arg("new_npix"),                         //
      nb::arg("weighted_counts").noconvert(),      //
      nb::arg("weighted_sin_sq").noconvert(),      //
      nb::arg("weighted_cos_sq").noconvert(),      //
      nb::arg("weighted_sincos").noconvert(),      //
      nb::arg("weighted_sin").noconvert(),         //
      nb::arg("weighted_cos").noconvert(),         //
      nb::arg("one_over_determinant").noconvert(), //
      nb::arg("vec").noconvert(),                  //
      nb::arg("prod").noconvert()                  //
  );
}

NB_MODULE(BlkDiagPrecondLO_tools, m) {
  m.doc() = "BlkDiagPrecondLO_tools";

  register_BlkDiagPrecondLO<double, nb::device::cpu>(m);
  register_BlkDiagPrecondLO<float, nb::device::cpu>(m);
}
