#include <functional>
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "mpi_utils.hpp"

namespace py = pybind11;

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

template <template <typename, int = py::array::c_style> class buffer_t,
          typename dint, typename dfloat>
std::function<void(                      //
    const ssize_t nsamples,              //
    const buffer_t<dint> pointings,      //
    const buffer_t<bool> pointings_flag, //
    const buffer_t<dfloat> vec,          //
    buffer_t<dfloat> prod                //
    )>
    numpy_bind_PLO_mult_I =             //
    [](const ssize_t nsamples,          //
       const py::buffer pointings,      //
       const py::buffer pointings_flag, //
       const py::buffer vec,            //
       py::buffer prod                  //
    ) {
      py::buffer_info pointings_info = pointings.request();
      py::buffer_info pointings_flag_info = pointings_flag.request();
      py::buffer_info vec_info = vec.request();
      py::buffer_info prod_info = prod.request();

      const dint *pointings_ptr =
          reinterpret_cast<const dint *>(pointings_info.ptr);
      const bool *pointings_flag_ptr =
          reinterpret_cast<const bool *>(pointings_flag_info.ptr);
      const dfloat *vec_ptr = reinterpret_cast<const dfloat *>(vec_info.ptr);
      dfloat *prod_ptr = reinterpret_cast<dfloat *>(prod_info.ptr);

      PLO_mult_I(             //
          nsamples,           //
          pointings_ptr,      //
          pointings_flag_ptr, //
          vec_ptr,            //
          prod_ptr            //
      );

      return;
    }; // numpy_bind_PLO_mult_I()

template <template <typename, int = py::array::c_style> class buffer_t,
          typename dint, typename dfloat>
std::function<void(                      //
    const ssize_t new_npix,              //
    const ssize_t nsamples,              //
    const buffer_t<dint> pointings,      //
    const buffer_t<bool> pointings_flag, //
    const buffer_t<dfloat> vec,          //
    buffer_t<dfloat> prod,               //
    const py::object mpi4py_comm         //
    )>
    numpy_bind_PLO_rmult_I =            //
    [](const ssize_t new_npix,          //
       const ssize_t nsamples,          //
       const py::buffer pointings,      //
       const py::buffer pointings_flag, //
       const py::buffer vec,            //
       py::buffer prod,                 //
       const py::object mpi4py_comm     //
    ) {
      py::buffer_info pointings_info = pointings.request();
      py::buffer_info pointings_flag_info = pointings_flag.request();
      py::buffer_info vec_info = vec.request();
      py::buffer_info prod_info = prod.request();

      const dint *pointings_ptr =
          reinterpret_cast<const dint *>(pointings_info.ptr);
      const bool *pointings_flag_ptr =
          reinterpret_cast<const bool *>(pointings_flag_info.ptr);
      const dfloat *vec_ptr = reinterpret_cast<const dfloat *>(vec_info.ptr);
      dfloat *prod_ptr = reinterpret_cast<dfloat *>(prod_info.ptr);

      const MPI_Comm comm =
          (reinterpret_cast<const PyMPICommObject *>(mpi4py_comm.ptr()))
              ->ob_mpi;

      PLO_rmult_I(            //
          new_npix,           //
          nsamples,           //
          pointings_ptr,      //
          pointings_flag_ptr, //
          vec_ptr,            //
          prod_ptr,           //
          comm                //
      );

      return;
    }; // numpy_bind_PLO_rmult_I()

template <template <typename, int = py::array::c_style> class buffer_t,
          typename dint, typename dfloat>
std::function<void(                      //
    const ssize_t nsamples,              //
    const buffer_t<dint> pointings,      //
    const buffer_t<bool> pointings_flag, //
    const buffer_t<dfloat> sin2phi,      //
    const buffer_t<dfloat> cos2phi,      //
    const buffer_t<dfloat> vec,          //
    buffer_t<dfloat> prod                //
    )>
    numpy_bind_PLO_mult_QU =            //
    [](const ssize_t nsamples,          //
       const py::buffer pointings,      //
       const py::buffer pointings_flag, //
       const py::buffer sin2phi,        //
       const py::buffer cos2phi,        //
       const py::buffer vec,            //
       py::buffer prod                  //
    ) {
      py::buffer_info pointings_info = pointings.request();
      py::buffer_info pointings_flag_info = pointings_flag.request();
      py::buffer_info sin2phi_info = sin2phi.request();
      py::buffer_info cos2phi_info = cos2phi.request();
      py::buffer_info vec_info = vec.request();
      py::buffer_info prod_info = prod.request();

      const dint *pointings_ptr =
          reinterpret_cast<const dint *>(pointings_info.ptr);
      const bool *pointings_flag_ptr =
          reinterpret_cast<const bool *>(pointings_flag_info.ptr);
      const dfloat *sin2phi_ptr =
          reinterpret_cast<const dfloat *>(sin2phi_info.ptr);
      const dfloat *cos2phi_ptr =
          reinterpret_cast<const dfloat *>(cos2phi_info.ptr);
      const dfloat *vec_ptr = reinterpret_cast<const dfloat *>(vec_info.ptr);
      dfloat *prod_ptr = reinterpret_cast<dfloat *>(prod_info.ptr);

      PLO_mult_QU(            //
          nsamples,           //
          pointings_ptr,      //
          pointings_flag_ptr, //
          sin2phi_ptr,        //
          cos2phi_ptr,        //
          vec_ptr,            //
          prod_ptr            //
      );

      return;
    }; // numpy_bind_PLO_mult_QU()

template <template <typename, int = py::array::c_style> class buffer_t,
          typename dint, typename dfloat>
std::function<void(                      //
    const ssize_t new_npix,              //
    const ssize_t nsamples,              //
    const buffer_t<dint> pointings,      //
    const buffer_t<bool> pointings_flag, //
    const buffer_t<dfloat> sin2phi,      //
    const buffer_t<dfloat> cos2phi,      //
    const buffer_t<dfloat> vec,          //
    buffer_t<dfloat> prod,               //
    const py::object mpi4py_comm         //
    )>
    numpy_bind_PLO_rmult_QU =           //
    [](const ssize_t new_npix,          //
       const ssize_t nsamples,          //
       const py::buffer pointings,      //
       const py::buffer pointings_flag, //
       const py::buffer sin2phi,        //
       const py::buffer cos2phi,        //
       const py::buffer vec,            //
       py::buffer prod,                 //
       const py::object mpi4py_comm     //
    ) {
      py::buffer_info pointings_info = pointings.request();
      py::buffer_info pointings_flag_info = pointings_flag.request();
      py::buffer_info sin2phi_info = sin2phi.request();
      py::buffer_info cos2phi_info = cos2phi.request();
      py::buffer_info vec_info = vec.request();
      py::buffer_info prod_info = prod.request();

      const dint *pointings_ptr =
          reinterpret_cast<const dint *>(pointings_info.ptr);
      const bool *pointings_flag_ptr =
          reinterpret_cast<const bool *>(pointings_flag_info.ptr);
      const dfloat *sin2phi_ptr =
          reinterpret_cast<const dfloat *>(sin2phi_info.ptr);
      const dfloat *cos2phi_ptr =
          reinterpret_cast<const dfloat *>(cos2phi_info.ptr);
      const dfloat *vec_ptr = reinterpret_cast<const dfloat *>(vec_info.ptr);
      dfloat *prod_ptr = reinterpret_cast<dfloat *>(prod_info.ptr);

      const MPI_Comm comm =
          (reinterpret_cast<const PyMPICommObject *>(mpi4py_comm.ptr()))
              ->ob_mpi;

      PLO_rmult_QU(           //
          new_npix,           //
          nsamples,           //
          pointings_ptr,      //
          pointings_flag_ptr, //
          sin2phi_ptr,        //
          cos2phi_ptr,        //
          vec_ptr,            //
          prod_ptr,           //
          comm                //
      );

      return;
    }; // numpy_bind_PLO_rmult_QU()

template <template <typename, int = py::array::c_style> class buffer_t,
          typename dint, typename dfloat>
std::function<void(                      //
    const ssize_t nsamples,              //
    const buffer_t<dint> pointings,      //
    const buffer_t<bool> pointings_flag, //
    const buffer_t<dfloat> sin2phi,      //
    const buffer_t<dfloat> cos2phi,      //
    const buffer_t<dfloat> vec,          //
    buffer_t<dfloat> prod                //
    )>
    numpy_bind_PLO_mult_IQU =           //
    [](const ssize_t nsamples,          //
       const py::buffer pointings,      //
       const py::buffer pointings_flag, //
       const py::buffer sin2phi,        //
       const py::buffer cos2phi,        //
       const py::buffer vec,            //
       py::buffer prod                  //
    ) {
      py::buffer_info pointings_info = pointings.request();
      py::buffer_info pointings_flag_info = pointings_flag.request();
      py::buffer_info sin2phi_info = sin2phi.request();
      py::buffer_info cos2phi_info = cos2phi.request();
      py::buffer_info vec_info = vec.request();
      py::buffer_info prod_info = prod.request();

      const dint *pointings_ptr =
          reinterpret_cast<const dint *>(pointings_info.ptr);
      const bool *pointings_flag_ptr =
          reinterpret_cast<const bool *>(pointings_flag_info.ptr);
      const dfloat *sin2phi_ptr =
          reinterpret_cast<const dfloat *>(sin2phi_info.ptr);
      const dfloat *cos2phi_ptr =
          reinterpret_cast<const dfloat *>(cos2phi_info.ptr);
      const dfloat *vec_ptr = reinterpret_cast<const dfloat *>(vec_info.ptr);
      dfloat *prod_ptr = reinterpret_cast<dfloat *>(prod_info.ptr);

      PLO_mult_IQU(           //
          nsamples,           //
          pointings_ptr,      //
          pointings_flag_ptr, //
          sin2phi_ptr,        //
          cos2phi_ptr,        //
          vec_ptr,            //
          prod_ptr            //
      );

      return;
    }; // numpy_bind_PLO_mult_IQU()

template <template <typename, int = py::array::c_style> class buffer_t,
          typename dint, typename dfloat>
std::function<void(                      //
    const ssize_t new_npix,              //
    const ssize_t nsamples,              //
    const buffer_t<dint> pointings,      //
    const buffer_t<bool> pointings_flag, //
    const buffer_t<dfloat> sin2phi,      //
    const buffer_t<dfloat> cos2phi,      //
    const buffer_t<dfloat> vec,          //
    buffer_t<dfloat> prod,               //
    const py::object mpi4py_comm         //
    )>
    numpy_bind_PLO_rmult_IQU =          //
    [](const ssize_t new_npix,          //
       const ssize_t nsamples,          //
       const py::buffer pointings,      //
       const py::buffer pointings_flag, //
       const py::buffer sin2phi,        //
       const py::buffer cos2phi,        //
       const py::buffer vec,            //
       py::buffer prod,                 //
       const py::object mpi4py_comm     //
    ) {
      py::buffer_info pointings_info = pointings.request();
      py::buffer_info pointings_flag_info = pointings_flag.request();
      py::buffer_info sin2phi_info = sin2phi.request();
      py::buffer_info cos2phi_info = cos2phi.request();
      py::buffer_info vec_info = vec.request();
      py::buffer_info prod_info = prod.request();

      const dint *pointings_ptr =
          reinterpret_cast<const dint *>(pointings_info.ptr);
      const bool *pointings_flag_ptr =
          reinterpret_cast<const bool *>(pointings_flag_info.ptr);
      const dfloat *sin2phi_ptr =
          reinterpret_cast<const dfloat *>(sin2phi_info.ptr);
      const dfloat *cos2phi_ptr =
          reinterpret_cast<const dfloat *>(cos2phi_info.ptr);
      const dfloat *vec_ptr = reinterpret_cast<const dfloat *>(vec_info.ptr);
      dfloat *prod_ptr = reinterpret_cast<dfloat *>(prod_info.ptr);

      const MPI_Comm comm =
          (reinterpret_cast<const PyMPICommObject *>(mpi4py_comm.ptr()))
              ->ob_mpi;

      PLO_rmult_IQU(          //
          new_npix,           //
          nsamples,           //
          pointings_ptr,      //
          pointings_flag_ptr, //
          sin2phi_ptr,        //
          cos2phi_ptr,        //
          vec_ptr,            //
          prod_ptr,           //
          comm                //
      );

      return;
    }; // numpy_bind_PLO_rmult_IQU()

PYBIND11_MODULE(PointingLO_tools, m) {
  m.doc() = "PointingLO_tools";
  m.def("PLO_mult_I", numpy_bind_PLO_mult_I<py::array_t, int32_t, float>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("PLO_mult_I", numpy_bind_PLO_mult_I<py::array_t, int32_t, double>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("PLO_mult_I", numpy_bind_PLO_mult_I<py::array_t, int64_t, float>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("PLO_mult_I", numpy_bind_PLO_mult_I<py::array_t, int64_t, double>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("PLO_mult_QU", numpy_bind_PLO_mult_QU<py::array_t, int32_t, float>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("PLO_mult_QU", numpy_bind_PLO_mult_QU<py::array_t, int32_t, double>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("PLO_mult_QU", numpy_bind_PLO_mult_QU<py::array_t, int64_t, float>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("PLO_mult_QU", numpy_bind_PLO_mult_QU<py::array_t, int64_t, double>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("PLO_mult_IQU", numpy_bind_PLO_mult_IQU<py::array_t, int32_t, float>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("PLO_mult_IQU", numpy_bind_PLO_mult_IQU<py::array_t, int32_t, double>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("PLO_mult_IQU", numpy_bind_PLO_mult_IQU<py::array_t, int64_t, float>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("PLO_mult_IQU", numpy_bind_PLO_mult_IQU<py::array_t, int64_t, double>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );

  m.def("PLO_rmult_I", numpy_bind_PLO_rmult_I<py::array_t, int32_t, float>,
        py::arg("new_npix"),                   //
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert(),           //
        py::arg("comm").noconvert()            //
  );
  m.def("PLO_rmult_I", numpy_bind_PLO_rmult_I<py::array_t, int32_t, double>,
        py::arg("new_npix"),                   //
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert(),           //
        py::arg("comm").noconvert()            //
  );
  m.def("PLO_rmult_I", numpy_bind_PLO_rmult_I<py::array_t, int64_t, float>,
        py::arg("new_npix"),                   //
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert(),           //
        py::arg("comm").noconvert()            //
  );
  m.def("PLO_rmult_I", numpy_bind_PLO_rmult_I<py::array_t, int64_t, double>,
        py::arg("new_npix"),                   //
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert(),           //
        py::arg("comm").noconvert()            //
  );
  m.def("PLO_rmult_QU", numpy_bind_PLO_rmult_QU<py::array_t, int32_t, float>,
        py::arg("new_npix"),                   //
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert(),           //
        py::arg("comm").noconvert()            //
  );
  m.def("PLO_rmult_QU", numpy_bind_PLO_rmult_QU<py::array_t, int32_t, double>,
        py::arg("new_npix"),                   //
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert(),           //
        py::arg("comm").noconvert()            //
  );
  m.def("PLO_rmult_QU", numpy_bind_PLO_rmult_QU<py::array_t, int64_t, float>,
        py::arg("new_npix"),                   //
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert(),           //
        py::arg("comm").noconvert()            //
  );
  m.def("PLO_rmult_QU", numpy_bind_PLO_rmult_QU<py::array_t, int64_t, double>,
        py::arg("new_npix"),                   //
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert(),           //
        py::arg("comm").noconvert()            //
  );
  m.def("PLO_rmult_IQU", numpy_bind_PLO_rmult_IQU<py::array_t, int32_t, float>,
        py::arg("new_npix"),                   //
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert(),           //
        py::arg("comm").noconvert()            //
  );
  m.def("PLO_rmult_IQU", numpy_bind_PLO_rmult_IQU<py::array_t, int32_t, double>,
        py::arg("new_npix"),                   //
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert(),           //
        py::arg("comm").noconvert()            //
  );
  m.def("PLO_rmult_IQU", numpy_bind_PLO_rmult_IQU<py::array_t, int64_t, float>,
        py::arg("new_npix"),                   //
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert(),           //
        py::arg("comm").noconvert()            //
  );
  m.def("PLO_rmult_IQU", numpy_bind_PLO_rmult_IQU<py::array_t, int64_t, double>,
        py::arg("new_npix"),                   //
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert(),           //
        py::arg("comm").noconvert()            //
  );
}
