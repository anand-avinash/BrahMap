#include <functional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename dint, typename dfloat>
void mult_I(                    //
    const ssize_t nsamples,     //
    const dint *pointings,      //
    const bool *pointings_flag, //
    const bool *pixel_flag,     //
    const dfloat *vec,          //
    dfloat *prod                //
) {

  for (ssize_t idx = 0; idx < nsamples; ++idx) {

    dint pixel = pointings[idx];
    bool pixflag = pixel_flag[pixel];

    // Here  we are multiplying `pixflag` to the pointings to prevent out of
    // range errors. Note that `pointings` array contains pathological pixels as
    // well. We don't want to look for them. While `pointings` array contains
    // number from 0 to `npix`, the size of `vec` array is only `new_npix`. The
    // same is applicable in all the functions in this file.
    pixel *= pixflag;

    // Multiplying `pixflag` with `pointings_flag` just flags the pointing
    // samples corresponding to the pathological pixel.
    bool pointflag = pixflag && pointings_flag[idx];

    prod[idx] += pointflag * vec[pixel];
  } // for

  return;

} // mult_I()

template <typename dint, typename dfloat>
void rmult_I(                   //
    const ssize_t nsamples,     //
    const dint *pointings,      //
    const bool *pointings_flag, //
    const bool *pixel_flag,     //
    const dfloat *vec,          //
    dfloat *prod                //
) {

  for (ssize_t idx = 0; idx < nsamples; ++idx) {

    dint pixel = pointings[idx];
    bool pixflag = pixel_flag[pixel];

    // Note that the size of `prod` vector is `new_npix` whereas `pointings`
    // array contains numbers from 0 to `npix`.
    pixel *= pixflag;
    bool pointflag = pixflag && pointings_flag[idx];

    prod[pixel] += pointflag * vec[idx];
  } // for

  return;

} // rmult_I()

template <typename dint, typename dfloat>
void mult_QU(                   //
    const ssize_t nsamples,     //
    const dint *pointings,      //
    const bool *pointings_flag, //
    const bool *pixel_flag,     //
    const dfloat *sin2phi,      //
    const dfloat *cos2phi,      //
    const dfloat *vec,          //
    dfloat *prod                //
) {

  for (ssize_t idx = 0; idx < nsamples; ++idx) {

    dint pixel = pointings[idx];
    bool pixflag = pixel_flag[pixel];

    pixel *= pixflag;
    bool pointflag = pixflag && pointings_flag[idx];

    prod[idx] += pointflag * (vec[2 * pixel] * cos2phi[idx] +
                              vec[2 * pixel + 1] * sin2phi[idx]);
  } // for

  return;
} // mult_QU()

template <typename dint, typename dfloat>
void rmult_QU(                  //
    const ssize_t nsamples,     //
    const dint *pointings,      //
    const bool *pointings_flag, //
    const bool *pixel_flag,     //
    const dfloat *sin2phi,      //
    const dfloat *cos2phi,      //
    const dfloat *vec,          //
    dfloat *prod                //
) {

  for (ssize_t idx = 0; idx < nsamples; ++idx) {

    dint pixel = pointings[idx];
    bool pixflag = pixel_flag[pixel];

    pixel *= pixflag;
    bool pointflag = pixflag && pointings_flag[idx];

    prod[2 * pixel] += pointflag * vec[idx] * cos2phi[idx];
    prod[2 * pixel + 1] += pointflag * vec[idx] * sin2phi[idx];
  } // for

  return;
} // rmult_QU()

template <typename dint, typename dfloat>
void mult_IQU(                  //
    const ssize_t nsamples,     //
    const dint *pointings,      //
    const bool *pointings_flag, //
    const bool *pixel_flag,     //
    const dfloat *sin2phi,      //
    const dfloat *cos2phi,      //
    const dfloat *vec,          //
    dfloat *prod                //
) {

  for (ssize_t idx = 0; idx < nsamples; ++idx) {

    dint pixel = pointings[idx];
    bool pixflag = pixel_flag[pixel];

    pixel *= pixflag;
    bool pointflag = pixflag && pointings_flag[idx];

    prod[idx] +=
        pointflag * (vec[3 * pixel] + vec[3 * pixel + 1] * cos2phi[idx] +
                     vec[3 * pixel + 2] * sin2phi[idx]);
  } // for

  return;
} // mult_IQU()

template <typename dint, typename dfloat>
void rmult_IQU(                 //
    const ssize_t nsamples,     //
    const dint *pointings,      //
    const bool *pointings_flag, //
    const bool *pixel_flag,     //
    const dfloat *sin2phi,      //
    const dfloat *cos2phi,      //
    const dfloat *vec,          //
    dfloat *prod                //
) {

  for (ssize_t idx = 0; idx < nsamples; ++idx) {

    dint pixel = pointings[idx];
    bool pixflag = pixel_flag[pixel];

    pixel *= pixflag;
    bool pointflag = pixflag && pointings_flag[idx];

    prod[3 * pixel] += pointflag * vec[idx];
    prod[3 * pixel + 1] += pointflag * vec[idx] * cos2phi[idx];
    prod[3 * pixel + 2] += pointflag * vec[idx] * sin2phi[idx];
  } // for

  return;
} // rmult_IQU()

template <typename dint, typename dfloat>
std::function<void(                         //
    const ssize_t nsamples,                 //
    const py::array_t<dint> pointings,      //
    const py::array_t<bool> pointings_flag, //
    const py::array_t<bool> pixel_flag,     //
    const py::array_t<dfloat> vec,          //
    py::array_t<dfloat> prod                //
    )>
    numpy_bind_mult_I =                 //
    [](const ssize_t nsamples,          //
       const py::buffer pointings,      //
       const py::buffer pointings_flag, //
       const py::buffer pixel_flag,     //
       const py::buffer vec,            //
       py::buffer prod                  //
    ) {
      py::buffer_info pointings_info = pointings.request();
      py::buffer_info pointings_flag_info = pointings_flag.request();
      py::buffer_info pixel_flag_info = pixel_flag.request();
      py::buffer_info vec_info = vec.request();
      py::buffer_info prod_info = prod.request();

      const dint *pointings_ptr = reinterpret_cast<dint *>(pointings_info.ptr);
      const bool *pointings_flag_ptr =
          reinterpret_cast<bool *>(pointings_flag_info.ptr);
      const bool *pixel_flag_ptr =
          reinterpret_cast<bool *>(pixel_flag_info.ptr);
      const dfloat *vec_ptr = reinterpret_cast<dfloat *>(vec_info.ptr);
      dfloat *prod_ptr = reinterpret_cast<dfloat *>(prod_info.ptr);

      mult_I(                 //
          nsamples,           //
          pointings_ptr,      //
          pointings_flag_ptr, //
          pixel_flag_ptr,     //
          vec_ptr,            //
          prod_ptr            //
      );

      return;
    }; // numpy_bind_mult_I()

template <typename dint, typename dfloat>
std::function<void(                         //
    const ssize_t nsamples,                 //
    const py::array_t<dint> pointings,      //
    const py::array_t<bool> pointings_flag, //
    const py::array_t<bool> pixel_flag,     //
    const py::array_t<dfloat> vec,          //
    py::array_t<dfloat> prod                //
    )>
    numpy_bind_rmult_I =                //
    [](const ssize_t nsamples,          //
       const py::buffer pointings,      //
       const py::buffer pointings_flag, //
       const py::buffer pixel_flag,     //
       const py::buffer vec,            //
       py::buffer prod                  //
    ) {
      py::buffer_info pointings_info = pointings.request();
      py::buffer_info pointings_flag_info = pointings_flag.request();
      py::buffer_info pixel_flag_info = pixel_flag.request();
      py::buffer_info vec_info = vec.request();
      py::buffer_info prod_info = prod.request();

      const dint *pointings_ptr = reinterpret_cast<dint *>(pointings_info.ptr);
      const bool *pointings_flag_ptr =
          reinterpret_cast<bool *>(pointings_flag_info.ptr);
      const bool *pixel_flag_ptr =
          reinterpret_cast<bool *>(pixel_flag_info.ptr);
      const dfloat *vec_ptr = reinterpret_cast<dfloat *>(vec_info.ptr);
      dfloat *prod_ptr = reinterpret_cast<dfloat *>(prod_info.ptr);

      rmult_I(                //
          nsamples,           //
          pointings_ptr,      //
          pointings_flag_ptr, //
          pixel_flag_ptr,     //
          vec_ptr,            //
          prod_ptr            //
      );

      return;
    }; // numpy_bind_rmult_I()

template <typename dint, typename dfloat>
std::function<void(                         //
    const ssize_t nsamples,                 //
    const py::array_t<dint> pointings,      //
    const py::array_t<bool> pointings_flag, //
    const py::array_t<bool> pixel_flag,     //
    const py::array_t<dfloat> sin2phi,      //
    const py::array_t<dfloat> cos2phi,      //
    const py::array_t<dfloat> vec,          //
    py::array_t<dfloat> prod                //
    )>
    numpy_bind_mult_QU =                //
    [](const ssize_t nsamples,          //
       const py::buffer pointings,      //
       const py::buffer pointings_flag, //
       const py::buffer pixel_flag,     //
       const py::buffer sin2phi,        //
       const py::buffer cos2phi,        //
       const py::buffer vec,            //
       py::buffer prod                  //
    ) {
      py::buffer_info pointings_info = pointings.request();
      py::buffer_info pointings_flag_info = pointings_flag.request();
      py::buffer_info pixel_flag_info = pixel_flag.request();
      py::buffer_info sin2phi_info = sin2phi.request();
      py::buffer_info cos2phi_info = cos2phi.request();
      py::buffer_info vec_info = vec.request();
      py::buffer_info prod_info = prod.request();

      const dint *pointings_ptr = reinterpret_cast<dint *>(pointings_info.ptr);
      const bool *pointings_flag_ptr =
          reinterpret_cast<bool *>(pointings_flag_info.ptr);
      const bool *pixel_flag_ptr =
          reinterpret_cast<bool *>(pixel_flag_info.ptr);
      const dfloat *sin2phi_ptr = reinterpret_cast<dfloat *>(sin2phi_info.ptr);
      const dfloat *cos2phi_ptr = reinterpret_cast<dfloat *>(cos2phi_info.ptr);
      const dfloat *vec_ptr = reinterpret_cast<dfloat *>(vec_info.ptr);
      dfloat *prod_ptr = reinterpret_cast<dfloat *>(prod_info.ptr);

      mult_QU(                //
          nsamples,           //
          pointings_ptr,      //
          pointings_flag_ptr, //
          pixel_flag_ptr,     //
          sin2phi_ptr,        //
          cos2phi_ptr,        //
          vec_ptr,            //
          prod_ptr            //
      );

      return;
    }; // numpy_bind_mult_QU()

template <typename dint, typename dfloat>
std::function<void(                         //
    const ssize_t nsamples,                 //
    const py::array_t<dint> pointings,      //
    const py::array_t<bool> pointings_flag, //
    const py::array_t<bool> pixel_flag,     //
    const py::array_t<dfloat> sin2phi,      //
    const py::array_t<dfloat> cos2phi,      //
    const py::array_t<dfloat> vec,          //
    py::array_t<dfloat> prod                //
    )>
    numpy_bind_rmult_QU =               //
    [](const ssize_t nsamples,          //
       const py::buffer pointings,      //
       const py::buffer pointings_flag, //
       const py::buffer pixel_flag,     //
       const py::buffer sin2phi,        //
       const py::buffer cos2phi,        //
       const py::buffer vec,            //
       py::buffer prod                  //
    ) {
      py::buffer_info pointings_info = pointings.request();
      py::buffer_info pointings_flag_info = pointings_flag.request();
      py::buffer_info pixel_flag_info = pixel_flag.request();
      py::buffer_info sin2phi_info = sin2phi.request();
      py::buffer_info cos2phi_info = cos2phi.request();
      py::buffer_info vec_info = vec.request();
      py::buffer_info prod_info = prod.request();

      const dint *pointings_ptr = reinterpret_cast<dint *>(pointings_info.ptr);
      const bool *pointings_flag_ptr =
          reinterpret_cast<bool *>(pointings_flag_info.ptr);
      const bool *pixel_flag_ptr =
          reinterpret_cast<bool *>(pixel_flag_info.ptr);
      const dfloat *sin2phi_ptr = reinterpret_cast<dfloat *>(sin2phi_info.ptr);
      const dfloat *cos2phi_ptr = reinterpret_cast<dfloat *>(cos2phi_info.ptr);
      const dfloat *vec_ptr = reinterpret_cast<dfloat *>(vec_info.ptr);
      dfloat *prod_ptr = reinterpret_cast<dfloat *>(prod_info.ptr);

      rmult_QU(               //
          nsamples,           //
          pointings_ptr,      //
          pointings_flag_ptr, //
          pixel_flag_ptr,     //
          sin2phi_ptr,        //
          cos2phi_ptr,        //
          vec_ptr,            //
          prod_ptr            //
      );

      return;
    }; // numpy_bind_rmult_QU()

template <typename dint, typename dfloat>
std::function<void(                         //
    const ssize_t nsamples,                 //
    const py::array_t<dint> pointings,      //
    const py::array_t<bool> pointings_flag, //
    const py::array_t<bool> pixel_flag,     //
    const py::array_t<dfloat> sin2phi,      //
    const py::array_t<dfloat> cos2phi,      //
    const py::array_t<dfloat> vec,          //
    py::array_t<dfloat> prod                //
    )>
    numpy_bind_mult_IQU =               //
    [](const ssize_t nsamples,          //
       const py::buffer pointings,      //
       const py::buffer pointings_flag, //
       const py::buffer pixel_flag,     //
       const py::buffer sin2phi,        //
       const py::buffer cos2phi,        //
       const py::buffer vec,            //
       py::buffer prod                  //
    ) {
      py::buffer_info pointings_info = pointings.request();
      py::buffer_info pointings_flag_info = pointings_flag.request();
      py::buffer_info pixel_flag_info = pixel_flag.request();
      py::buffer_info sin2phi_info = sin2phi.request();
      py::buffer_info cos2phi_info = cos2phi.request();
      py::buffer_info vec_info = vec.request();
      py::buffer_info prod_info = prod.request();

      const dint *pointings_ptr = reinterpret_cast<dint *>(pointings_info.ptr);
      const bool *pointings_flag_ptr =
          reinterpret_cast<bool *>(pointings_flag_info.ptr);
      const bool *pixel_flag_ptr =
          reinterpret_cast<bool *>(pixel_flag_info.ptr);
      const dfloat *sin2phi_ptr = reinterpret_cast<dfloat *>(sin2phi_info.ptr);
      const dfloat *cos2phi_ptr = reinterpret_cast<dfloat *>(cos2phi_info.ptr);
      const dfloat *vec_ptr = reinterpret_cast<dfloat *>(vec_info.ptr);
      dfloat *prod_ptr = reinterpret_cast<dfloat *>(prod_info.ptr);

      mult_IQU(               //
          nsamples,           //
          pointings_ptr,      //
          pointings_flag_ptr, //
          pixel_flag_ptr,     //
          sin2phi_ptr,        //
          cos2phi_ptr,        //
          vec_ptr,            //
          prod_ptr            //
      );

      return;
    }; // numpy_bind_mult_IQU()

template <typename dint, typename dfloat>
std::function<void(                         //
    const ssize_t nsamples,                 //
    const py::array_t<dint> pointings,      //
    const py::array_t<bool> pointings_flag, //
    const py::array_t<bool> pixel_flag,     //
    const py::array_t<dfloat> sin2phi,      //
    const py::array_t<dfloat> cos2phi,      //
    const py::array_t<dfloat> vec,          //
    py::array_t<dfloat> prod                //
    )>
    numpy_bind_rmult_IQU =              //
    [](const ssize_t nsamples,          //
       const py::buffer pointings,      //
       const py::buffer pointings_flag, //
       const py::buffer pixel_flag,     //
       const py::buffer sin2phi,        //
       const py::buffer cos2phi,        //
       const py::buffer vec,            //
       py::buffer prod                  //
    ) {
      py::buffer_info pointings_info = pointings.request();
      py::buffer_info pointings_flag_info = pointings_flag.request();
      py::buffer_info pixel_flag_info = pixel_flag.request();
      py::buffer_info sin2phi_info = sin2phi.request();
      py::buffer_info cos2phi_info = cos2phi.request();
      py::buffer_info vec_info = vec.request();
      py::buffer_info prod_info = prod.request();

      const dint *pointings_ptr = reinterpret_cast<dint *>(pointings_info.ptr);
      const bool *pointings_flag_ptr =
          reinterpret_cast<bool *>(pointings_flag_info.ptr);
      const bool *pixel_flag_ptr =
          reinterpret_cast<bool *>(pixel_flag_info.ptr);
      const dfloat *sin2phi_ptr = reinterpret_cast<dfloat *>(sin2phi_info.ptr);
      const dfloat *cos2phi_ptr = reinterpret_cast<dfloat *>(cos2phi_info.ptr);
      const dfloat *vec_ptr = reinterpret_cast<dfloat *>(vec_info.ptr);
      dfloat *prod_ptr = reinterpret_cast<dfloat *>(prod_info.ptr);

      rmult_IQU(              //
          nsamples,           //
          pointings_ptr,      //
          pointings_flag_ptr, //
          pixel_flag_ptr,     //
          sin2phi_ptr,        //
          cos2phi_ptr,        //
          vec_ptr,            //
          prod_ptr            //
      );

      return;
    }; // numpy_bind_rmult_IQU()

PYBIND11_MODULE(PointingLO_tools, m) {
  m.doc() = "PointingLO_tools";
  m.def("mult_I", numpy_bind_mult_I<int32_t, float>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("mult_I", numpy_bind_mult_I<int32_t, double>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("mult_I", numpy_bind_mult_I<int64_t, float>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("mult_I", numpy_bind_mult_I<int64_t, double>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("mult_QU", numpy_bind_mult_QU<int32_t, float>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("mult_QU", numpy_bind_mult_QU<int32_t, double>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("mult_QU", numpy_bind_mult_QU<int64_t, float>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("mult_QU", numpy_bind_mult_QU<int64_t, double>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("mult_IQU", numpy_bind_mult_IQU<int32_t, float>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("mult_IQU", numpy_bind_mult_IQU<int32_t, double>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("mult_IQU", numpy_bind_mult_IQU<int64_t, float>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("mult_IQU", numpy_bind_mult_IQU<int64_t, double>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );

  m.def("rmult_I", numpy_bind_rmult_I<int32_t, float>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("rmult_I", numpy_bind_rmult_I<int32_t, double>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("rmult_I", numpy_bind_rmult_I<int64_t, float>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("rmult_I", numpy_bind_rmult_I<int64_t, double>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("rmult_QU", numpy_bind_rmult_QU<int32_t, float>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("rmult_QU", numpy_bind_rmult_QU<int32_t, double>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("rmult_QU", numpy_bind_rmult_QU<int64_t, float>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("rmult_QU", numpy_bind_rmult_QU<int64_t, double>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("rmult_IQU", numpy_bind_rmult_IQU<int32_t, float>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("rmult_IQU", numpy_bind_rmult_IQU<int32_t, double>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("rmult_IQU", numpy_bind_rmult_IQU<int64_t, float>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
  m.def("rmult_IQU", numpy_bind_rmult_IQU<int64_t, double>,
        py::arg("nsamples"),                   //
        py::arg("pointings").noconvert(),      //
        py::arg("pointings_flag").noconvert(), //
        py::arg("pixel_flag").noconvert(),     //
        py::arg("sin2phi").noconvert(),        //
        py::arg("cos2phi").noconvert(),        //
        py::arg("vec").noconvert(),            //
        py::arg("prod").noconvert()            //
  );
}
