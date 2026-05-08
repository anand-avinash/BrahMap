#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#ifndef _DISABLE_OMP
#include <omp.h>
#endif

#include "mpi_utils.hpp"

namespace nb = nanobind;

template <typename dint, typename dfloat>
void PLO_mult_I(                           //
    const ssize_t nsamples,                //
    const dint *__restrict pointings,      //
    const bool *__restrict pointings_flag, //
    const dfloat *__restrict vec,          //
    dfloat *__restrict prod                //
) {

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < nsamples; ++idx) {

    dint pixel = pointings[idx];
    bool pointflag = pointings_flag[idx];

    prod[idx] += pointflag * vec[pixel];
  } // for

  return;

} // PLO_mult_I()

template <typename dint, typename dfloat>
void PLO_rmult_I(                          //
    const ssize_t new_npix,                //
    const ssize_t nsamples,                //
    const dint *__restrict pointings,      //
    const bool *__restrict pointings_flag, //
    const dfloat *__restrict vec,          //
    dfloat *__restrict prod,               //
    const MPI_Comm comm                    //
) {

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < nsamples; ++idx) {

    dint pixel = pointings[idx];
    bool pointflag = pointings_flag[idx];

    dfloat product = pointflag * vec[idx];

#pragma omp atomic update
    prod[pixel] += product;
  } // for

  MPI_Allreduce(MPI_IN_PLACE, prod, new_npix, mpi_get_type<dfloat>(), MPI_SUM,
                comm);

  return;

} // PLO_rmult_I()

template <typename dint, typename dfloat>
void PLO_mult_QU(                          //
    const ssize_t nsamples,                //
    const dint *__restrict pointings,      //
    const bool *__restrict pointings_flag, //
    const dfloat *__restrict sin2phi,      //
    const dfloat *__restrict cos2phi,      //
    const dfloat *__restrict vec,          //
    dfloat *__restrict prod                //
) {

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < nsamples; ++idx) {

    dint pixel = pointings[idx];
    bool pointflag = pointings_flag[idx];

    prod[idx] += pointflag * (vec[2 * pixel] * cos2phi[idx] +
                              vec[2 * pixel + 1] * sin2phi[idx]);
  } // for

  return;
} // PLO_mult_QU()

template <typename dint, typename dfloat>
void PLO_rmult_QU(                         //
    const ssize_t new_npix,                //
    const ssize_t nsamples,                //
    const dint *__restrict pointings,      //
    const bool *__restrict pointings_flag, //
    const dfloat *__restrict sin2phi,      //
    const dfloat *__restrict cos2phi,      //
    const dfloat *__restrict vec,          //
    dfloat *__restrict prod,               //
    const MPI_Comm comm                    //
) {

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < nsamples; ++idx) {

    dint pixel = pointings[idx];
    bool pointflag = pointings_flag[idx];

    dfloat product_1 = pointflag * vec[idx] * cos2phi[idx];
    dfloat product_2 = pointflag * vec[idx] * sin2phi[idx];

#pragma omp atomic update
    prod[2 * pixel] += product_1;
#pragma omp atomic update
    prod[2 * pixel + 1] += product_2;

  } // for

  MPI_Allreduce(MPI_IN_PLACE, prod, 2 * new_npix, mpi_get_type<dfloat>(),
                MPI_SUM, comm);

  return;
} // PLO_rmult_QU()

template <typename dint, typename dfloat>
void PLO_mult_IQU(                         //
    const ssize_t nsamples,                //
    const dint *__restrict pointings,      //
    const bool *__restrict pointings_flag, //
    const dfloat *__restrict sin2phi,      //
    const dfloat *__restrict cos2phi,      //
    const dfloat *__restrict vec,          //
    dfloat *__restrict prod                //
) {

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < nsamples; ++idx) {

    dint pixel = pointings[idx];
    bool pointflag = pointings_flag[idx];

    prod[idx] +=
        pointflag * (vec[3 * pixel] + vec[3 * pixel + 1] * cos2phi[idx] +
                     vec[3 * pixel + 2] * sin2phi[idx]);
  } // for

  return;
} // PLO_mult_IQU()

template <typename dint, typename dfloat>
void PLO_rmult_IQU(                        //
    const ssize_t new_npix,                //
    const ssize_t nsamples,                //
    const dint *__restrict pointings,      //
    const bool *__restrict pointings_flag, //
    const dfloat *__restrict sin2phi,      //
    const dfloat *__restrict cos2phi,      //
    const dfloat *__restrict vec,          //
    dfloat *__restrict prod,               //
    const MPI_Comm comm                    //
) {

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < nsamples; ++idx) {

    dint pixel = pointings[idx];
    bool pointflag = pointings_flag[idx];

    dfloat product_1 = pointflag * vec[idx];
    dfloat product_2 = pointflag * vec[idx] * cos2phi[idx];
    dfloat product_3 = pointflag * vec[idx] * sin2phi[idx];

#pragma omp atomic update
    prod[3 * pixel] += product_1;
#pragma omp atomic update
    prod[3 * pixel + 1] += product_2;
#pragma omp atomic update
    prod[3 * pixel + 2] += product_3;
  } // for

  MPI_Allreduce(MPI_IN_PLACE, prod, 3 * new_npix, mpi_get_type<dfloat>(),
                MPI_SUM, comm);

  return;
} // PLO_rmult_IQU()

template <typename dint, typename dfloat, typename device> //
void register_PointingLO(nb::module_ &m) {
  using arr_dint = nb::ndarray<dint, nb::ndim<1>, device, nb::c_contig>;
  using arr_dfloat = nb::ndarray<dfloat, nb::ndim<1>, device, nb::c_contig>;
  using arr_bool = nb::ndarray<bool, nb::ndim<1>, device, nb::c_contig>;

  auto get_comm = [](const nb::object &mpi4py_comm) -> MPI_Comm {
    return (reinterpret_cast<const PyMPICommObject *>(mpi4py_comm.ptr()))
        ->ob_mpi;
  };

  m.def(
      "PLO_mult_I",                      //
      [](                                //
          const ssize_t nsamples,        //
          const arr_dint pointings,      //
          const arr_bool pointings_flag, //
          const arr_dfloat vec,          //
          arr_dfloat prod                //
      ) {
        PLO_mult_I(                //
            nsamples,              //
            pointings.data(),      //
            pointings_flag.data(), //
            vec.data(),            //
            prod.data()            //
        );
      },
      nb::arg("nsamples"),                   //
      nb::arg("pointings").noconvert(),      //
      nb::arg("pointings_flag").noconvert(), //
      nb::arg("vec").noconvert(),            //
      nb::arg("prod").noconvert()            //
  );

  m.def(
      "PLO_rmult_I",                     //
      [get_comm](                        //
          const ssize_t new_npix,        //
          const ssize_t nsamples,        //
          const arr_dint pointings,      //
          const arr_bool pointings_flag, //
          const arr_dfloat vec,          //
          arr_dfloat prod,               //
          const nb::object mpi4py_comm   //
      ) {
        PLO_rmult_I(               //
            new_npix,              //
            nsamples,              //
            pointings.data(),      //
            pointings_flag.data(), //
            vec.data(),            //
            prod.data(),           //
            get_comm(mpi4py_comm)  //
        );
      },
      nb::arg("new_npix"),                   //
      nb::arg("nsamples"),                   //
      nb::arg("pointings").noconvert(),      //
      nb::arg("pointings_flag").noconvert(), //
      nb::arg("vec").noconvert(),            //
      nb::arg("prod").noconvert(),           //
      nb::arg("comm").noconvert()            //
  );

  m.def(
      "PLO_mult_QU",                     //
      [](                                //
          const ssize_t nsamples,        //
          const arr_dint pointings,      //
          const arr_bool pointings_flag, //
          const arr_dfloat sin2phi,      //
          const arr_dfloat cos2phi,      //
          const arr_dfloat vec,          //
          arr_dfloat prod                //
      ) {
        PLO_mult_QU(               //
            nsamples,              //
            pointings.data(),      //
            pointings_flag.data(), //
            sin2phi.data(),        //
            cos2phi.data(),        //
            vec.data(),            //
            prod.data()            //
        );
      },
      nb::arg("nsamples"),                   //
      nb::arg("pointings").noconvert(),      //
      nb::arg("pointings_flag").noconvert(), //
      nb::arg("sin2phi").noconvert(),        //
      nb::arg("cos2phi").noconvert(),        //
      nb::arg("vec").noconvert(),            //
      nb::arg("prod").noconvert()            //
  );

  m.def(
      "PLO_rmult_QU",                    //
      [get_comm](                        //
          const ssize_t new_npix,        //
          const ssize_t nsamples,        //
          const arr_dint pointings,      //
          const arr_bool pointings_flag, //
          const arr_dfloat sin2phi,      //
          const arr_dfloat cos2phi,      //
          const arr_dfloat vec,          //
          arr_dfloat prod,               //
          const nb::object mpi4py_comm   //
      ) {
        PLO_rmult_QU(              //
            new_npix,              //
            nsamples,              //
            pointings.data(),      //
            pointings_flag.data(), //
            sin2phi.data(),        //
            cos2phi.data(),        //
            vec.data(),            //
            prod.data(),           //
            get_comm(mpi4py_comm)  //
        );
      },
      nb::arg("new_npix"),                   //
      nb::arg("nsamples"),                   //
      nb::arg("pointings").noconvert(),      //
      nb::arg("pointings_flag").noconvert(), //
      nb::arg("sin2phi").noconvert(),        //
      nb::arg("cos2phi").noconvert(),        //
      nb::arg("vec").noconvert(),            //
      nb::arg("prod").noconvert(),           //
      nb::arg("comm").noconvert()            //
  );

  m.def(
      "PLO_mult_IQU",                    //
      [](                                //
          const ssize_t nsamples,        //
          const arr_dint pointings,      //
          const arr_bool pointings_flag, //
          const arr_dfloat sin2phi,      //
          const arr_dfloat cos2phi,      //
          const arr_dfloat vec,          //
          arr_dfloat prod                //
      ) {
        PLO_mult_IQU(              //
            nsamples,              //
            pointings.data(),      //
            pointings_flag.data(), //
            sin2phi.data(),        //
            cos2phi.data(),        //
            vec.data(),            //
            prod.data()            //
        );
      },
      nb::arg("nsamples"),                   //
      nb::arg("pointings").noconvert(),      //
      nb::arg("pointings_flag").noconvert(), //
      nb::arg("sin2phi").noconvert(),        //
      nb::arg("cos2phi").noconvert(),        //
      nb::arg("vec").noconvert(),            //
      nb::arg("prod").noconvert()            //
  );

  m.def(
      "PLO_rmult_IQU",                   //
      [get_comm](                        //
          const ssize_t new_npix,        //
          const ssize_t nsamples,        //
          const arr_dint pointings,      //
          const arr_bool pointings_flag, //
          const arr_dfloat sin2phi,      //
          const arr_dfloat cos2phi,      //
          const arr_dfloat vec,          //
          arr_dfloat prod,               //
          const nb::object mpi4py_comm   //
      ) {
        PLO_rmult_IQU(             //
            new_npix,              //
            nsamples,              //
            pointings.data(),      //
            pointings_flag.data(), //
            sin2phi.data(),        //
            cos2phi.data(),        //
            vec.data(),            //
            prod.data(),           //
            get_comm(mpi4py_comm)  //
        );
      },
      nb::arg("new_npix"),                   //
      nb::arg("nsamples"),                   //
      nb::arg("pointings").noconvert(),      //
      nb::arg("pointings_flag").noconvert(), //
      nb::arg("sin2phi").noconvert(),        //
      nb::arg("cos2phi").noconvert(),        //
      nb::arg("vec").noconvert(),            //
      nb::arg("prod").noconvert(),           //
      nb::arg("comm").noconvert()            //
  );
}

NB_MODULE(PointingLO_tools, m) {
  m.doc() = "PointingLO_tools";

  register_PointingLO<int64_t, double, nb::device::cpu>(m);
  register_PointingLO<int32_t, double, nb::device::cpu>(m);
  register_PointingLO<int64_t, float, nb::device::cpu>(m);
  register_PointingLO<int32_t, float, nb::device::cpu>(m);
}
