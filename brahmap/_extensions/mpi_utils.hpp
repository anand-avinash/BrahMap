#ifndef _MPI_UTILS
#define _MPI_UTILS

#include <complex>
#include <cstddef>
#include <cstring>
#include <mpi.h>
#include <mpi4py/mpi4py.h>
#include <vector>

// The following function is taken from
// <https://gist.github.com/2b-t/50d85115db8b12ed263f8231abf07fa2>
template <typename T>
[[nodiscard]] constexpr MPI_Datatype mpi_get_type() noexcept {

  MPI_Datatype mpi_type = MPI_DATATYPE_NULL;

  if constexpr (std::is_same<T, char>::value) {
    mpi_type = MPI_CHAR;
  } else if constexpr (std::is_same<T, signed char>::value) {
    mpi_type = MPI_SIGNED_CHAR;
  } else if constexpr (std::is_same<T, unsigned char>::value) {
    mpi_type = MPI_UNSIGNED_CHAR;
  } else if constexpr (std::is_same<T, wchar_t>::value) {
    mpi_type = MPI_WCHAR;
  } else if constexpr (std::is_same<T, signed short>::value) {
    mpi_type = MPI_SHORT;
  } else if constexpr (std::is_same<T, unsigned short>::value) {
    mpi_type = MPI_UNSIGNED_SHORT;
  } else if constexpr (std::is_same<T, signed int>::value) {
    mpi_type = MPI_INT;
  } else if constexpr (std::is_same<T, unsigned int>::value) {
    mpi_type = MPI_UNSIGNED;
  } else if constexpr (std::is_same<T, signed long int>::value) {
    mpi_type = MPI_LONG;
  } else if constexpr (std::is_same<T, unsigned long int>::value) {
    mpi_type = MPI_UNSIGNED_LONG;
  } else if constexpr (std::is_same<T, signed long long int>::value) {
    mpi_type = MPI_LONG_LONG;
  } else if constexpr (std::is_same<T, unsigned long long int>::value) {
    mpi_type = MPI_UNSIGNED_LONG_LONG;
  } else if constexpr (std::is_same<T, float>::value) {
    mpi_type = MPI_FLOAT;
  } else if constexpr (std::is_same<T, double>::value) {
    mpi_type = MPI_DOUBLE;
  } else if constexpr (std::is_same<T, long double>::value) {
    mpi_type = MPI_LONG_DOUBLE;
  } else if constexpr (std::is_same<T, std::int8_t>::value) {
    mpi_type = MPI_INT8_T;
  } else if constexpr (std::is_same<T, std::int16_t>::value) {
    mpi_type = MPI_INT16_T;
  } else if constexpr (std::is_same<T, std::int32_t>::value) {
    mpi_type = MPI_INT32_T;
  } else if constexpr (std::is_same<T, std::int64_t>::value) {
    mpi_type = MPI_INT64_T;
  } else if constexpr (std::is_same<T, std::uint8_t>::value) {
    mpi_type = MPI_UINT8_T;
  } else if constexpr (std::is_same<T, std::uint16_t>::value) {
    mpi_type = MPI_UINT16_T;
  } else if constexpr (std::is_same<T, std::uint32_t>::value) {
    mpi_type = MPI_UINT32_T;
  } else if constexpr (std::is_same<T, std::uint64_t>::value) {
    mpi_type = MPI_UINT64_T;
  } else if constexpr (std::is_same<T, bool>::value) {
    mpi_type = MPI_C_BOOL;
  } else if constexpr (std::is_same<T, std::complex<float>>::value) {
    mpi_type = MPI_C_COMPLEX;
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    mpi_type = MPI_C_DOUBLE_COMPLEX;
  } else if constexpr (std::is_same<T, std::complex<long double>>::value) {
    mpi_type = MPI_C_LONG_DOUBLE_COMPLEX;
  } // if

  assert(mpi_type != MPI_DATATYPE_NULL);
  return mpi_type;

} // mpi_get_type()

// MPI shared memory allocator class
class SharedMemoryAllocator {
private:
  MPI_Comm _comm;
  int _rank;
  int _root;

  std::vector<MPI_Win> _windows;

public:
  // constructor
  SharedMemoryAllocator(MPI_Comm comm, int comm_root = 0)
      : _comm(comm), _root(comm_root) {
    MPI_Comm_rank(comm, &_rank);
  } // constructor

  // Disallow copying to prevent double-freeing MPI_Windows
  SharedMemoryAllocator(const SharedMemoryAllocator &) = delete;
  SharedMemoryAllocator &operator=(const SharedMemoryAllocator &) = delete;

  // Shared memory array allocator
  template <typename dtype> dtype *allocate(size_t size) {
    int dtype_size = sizeof(dtype);

    MPI_Aint bytes_to_allocate = (_rank == _root) ? size * dtype_size : 0;

    dtype *local_ptr = nullptr;
    MPI_Win win;
    MPI_Win_allocate_shared(bytes_to_allocate, dtype_size, MPI_INFO_NULL, _comm,
                            &local_ptr, &win);

    MPI_Aint segment_size;
    int disp;
    dtype *shared_buf = nullptr;

    // Query the root's memory space
    MPI_Win_shared_query(win, _root, &segment_size, &disp, &shared_buf);

    _windows.push_back(win);

    if (_rank == _root && shared_buf != nullptr) {
      std::memset(shared_buf, 0, size * dtype_size);
    } // if

    return shared_buf;

  } // allocate()

  // Expose the windows if the user needs to manually call MPI_Win_fence
  const std::vector<MPI_Win> &get_windows() const { return _windows; }

  // destructor
  ~SharedMemoryAllocator() {
    for (MPI_Win &win : _windows) {
      MPI_Win_free(&win);
    } // for
  }   // destructor

}; // SharedMemoryAllocator

#endif