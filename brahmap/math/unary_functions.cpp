#include <cmath>
#include <functional>
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Alias for the signature of a general unary function pointer.
// Here the unary function refers to a function that takes one
// input and returns one output.
template <typename dfloat> using dfunc = dfloat (*)(const dfloat);

// Templated function to call a general unary function `unary` over an array
template <typename dfloat, dfunc<dfloat> unary>
void execute(                     //
    const ssize_t size,           //
    const dfloat *__restrict vec, //
    dfloat *__restrict result     //
) {

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < size; ++idx) {
    result[idx] = unary(vec[idx]);
  } // for

  return;
} // execute()

template <template <typename, int = py::array::c_style> class buffer_t,
          typename dfloat,
          dfunc<dfloat> unary>
std::function<void(             //
    const ssize_t size,         //
    const buffer_t<dfloat> vec, //
    buffer_t<dfloat> result     //
    )>
    numpy_bind_unary =       //
    [](const ssize_t size,   //
       const py::buffer vec, //
       py::buffer result     //
    ) {
      py::buffer_info vec_info = vec.request();
      py::buffer_info result_info = result.request();

      const dfloat *vec_ptr = reinterpret_cast<const dfloat *>(vec_info.ptr);
      dfloat *result_ptr = reinterpret_cast<dfloat *>(result_info.ptr);

      execute<dfloat, unary>( //
          size,               //
          vec_ptr,            //
          result_ptr          //
      );

      return;
    }; // numpy_bind_unary()

PYBIND11_MODULE(unary_functions, m) {
  m.doc() = "unary_functions";

  // sin function
  m.def("sin", numpy_bind_unary<py::array_t, float, std::sin>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );
  m.def("sin", numpy_bind_unary<py::array_t, double, std::sin>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );

  // cos function
  m.def("cos", numpy_bind_unary<py::array_t, float, std::cos>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );
  m.def("cos", numpy_bind_unary<py::array_t, double, std::cos>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );

  // tan function
  m.def("tan", numpy_bind_unary<py::array_t, float, std::tan>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );
  m.def("tan", numpy_bind_unary<py::array_t, double, std::tan>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );

  // arcsin function
  m.def("arcsin", numpy_bind_unary<py::array_t, float, std::asin>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );
  m.def("arcsin", numpy_bind_unary<py::array_t, double, std::asin>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );

  // arccos function
  m.def("arccos", numpy_bind_unary<py::array_t, float, std::acos>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );
  m.def("arccos", numpy_bind_unary<py::array_t, double, std::acos>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );

  // arctan function
  m.def("arctan", numpy_bind_unary<py::array_t, float, std::atan>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );
  m.def("arctan", numpy_bind_unary<py::array_t, double, std::atan>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );

  // exp function: to compute e**x
  m.def("exp", numpy_bind_unary<py::array_t, float, std::exp>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );
  m.def("exp", numpy_bind_unary<py::array_t, double, std::exp>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );

  // exp2 function: to compute 2**x
  m.def("exp2", numpy_bind_unary<py::array_t, float, std::exp2>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );
  m.def("exp2", numpy_bind_unary<py::array_t, double, std::exp2>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );

  // log function: natural log
  m.def("log", numpy_bind_unary<py::array_t, float, std::log>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );
  m.def("log", numpy_bind_unary<py::array_t, double, std::log>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );

  // log2 function: log base-2
  m.def("log2", numpy_bind_unary<py::array_t, float, std::log2>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );
  m.def("log2", numpy_bind_unary<py::array_t, double, std::log2>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );

  // sqrt function: square root
  m.def("sqrt", numpy_bind_unary<py::array_t, float, std::sqrt>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );
  m.def("sqrt", numpy_bind_unary<py::array_t, double, std::sqrt>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );

  // cbrt function: cube root
  m.def("cbrt", numpy_bind_unary<py::array_t, float, std::cbrt>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );
  m.def("cbrt", numpy_bind_unary<py::array_t, double, std::cbrt>,
        py::arg("size"),              //
        py::arg("vec").noconvert(),   //
        py::arg("result").noconvert() //
  );
}
