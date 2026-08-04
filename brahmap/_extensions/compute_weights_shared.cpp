#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#ifndef _DISABLE_OMP
#include <omp.h>
#endif

#include "compute_weights_commons.cpp"
#include "mpi_utils.hpp"

namespace nb = nanobind;

///////////////////////////////
// Compute weights functions //
///////////////////////////////

template <typename dint, typename dfloat>
dint compute_weights_shmem_pol_I(            //
    const ssize_t npix,                      //
    const ssize_t nsamples,                  //
    const dint *__restrict pointings,        //
    const bool *__restrict pointings_flag,   //
    const dfloat *__restrict noise_weights,  //
    dint *__restrict node_hit_counts,        //
    MPI_Win &win_hit_counts,                 //
    dfloat *__restrict node_weighted_counts, //
    MPI_Win &win_weighted_counts,            //
    dint *__restrict observed_pixels,        //
    dint *__restrict __old2new_pixel,        //
    bool *__restrict pixel_flag,             //
    const ssize_t node_root,                 //
    const bool grp_reduce,                   //
    const MPI_Comm tree_grp_comm,            //
    const MPI_Comm tree_grp_root_comm,       //
    const MPI_Comm node_comm,                //
    const MPI_Comm node_root_comm            //
) {

  int tree_grp_rank, tree_grp_size, node_rank;
  MPI_Comm_rank(tree_grp_comm, &tree_grp_rank);
  MPI_Comm_size(tree_grp_comm, &tree_grp_size);
  MPI_Comm_rank(node_comm, &node_rank);

  dint *grp_hit_counts = nullptr;
  dfloat *grp_weighted_counts = nullptr;
  SharedMemoryAllocator *allocator = nullptr;

  if (tree_grp_size == 1) {
    grp_hit_counts = node_hit_counts;
    grp_weighted_counts = node_weighted_counts;
  } else {
    allocator = new SharedMemoryAllocator(tree_grp_comm, 0);
    grp_hit_counts = allocator->allocate<dint>(npix);
    grp_weighted_counts = allocator->allocate<dfloat>(npix);
  } // if

  if (allocator) {
    allocator->fence(0);
  } // if

  // Accumulation over group roots
  for (ssize_t idx = 0; idx < tree_grp_size; ++idx) {
    if (tree_grp_rank == idx) {

      accumulate_weights_pol_I<dint, dfloat>( //
          nsamples,                           //
          pointings,                          //
          pointings_flag,                     //
          noise_weights,                      //
          grp_hit_counts,                     //
          grp_weighted_counts                 //
      );

    } // if

    if (allocator) {
      allocator->fence(0);
    } // if
  }   // for

  // Group roots to node root reduction on each node
  // tree_grp_root_comm contains one root from each tree group communicator.
  // Since node_hit_counts is shared to all process on the node, it is
  // visible to all processes on the node, including the root of
  // tree_grp_root_comm. Therefore, it is safe and sufficient to provide the
  // 0 rank of tree_grp_root_comm as root argument of MPI_Reduce. This will
  // allow the reduction from the root of each tree group communicator to
  // the node root.
  // Also, for nproc_reduce == 1, grp_reduce = false

  if (grp_reduce) {
    if (tree_grp_root_comm != MPI_COMM_NULL) {
      MPI_Reduce(grp_hit_counts, node_hit_counts, npix, mpi_get_type<dint>(),
                 MPI_SUM, 0, tree_grp_root_comm);
      MPI_Reduce(grp_weighted_counts, node_weighted_counts, npix,
                 mpi_get_type<dfloat>(), MPI_SUM, 0, tree_grp_root_comm);
    } // if
  }   // if

  MPI_Win_fence(0, win_hit_counts);
  MPI_Win_fence(0, win_weighted_counts);

  if (allocator) {
    delete allocator;
  } // if

  // Allreduce sync across all node roots
  if (node_rank == node_root) {
    MPI_Allreduce(MPI_IN_PLACE, node_hit_counts, npix, mpi_get_type<dint>(),
                  MPI_SUM, node_root_comm);
    MPI_Allreduce(MPI_IN_PLACE, node_weighted_counts, npix,
                  mpi_get_type<dfloat>(), MPI_SUM, node_root_comm);
  } // if

  MPI_Win_fence(0, win_hit_counts);
  MPI_Win_fence(0, win_weighted_counts);

  dint new_npix;
  if (node_rank == node_root) {
    new_npix = 0;
    for (ssize_t idx = 0; idx < npix; ++idx) {
      if (node_hit_counts[idx] > 0) {
        observed_pixels[new_npix] = idx;
        __old2new_pixel[idx] = new_npix;
        pixel_flag[idx] = true;
        ++new_npix;
      } // if
    }   // for
  }     // if

  MPI_Bcast(&new_npix, 1, mpi_get_type<dint>(), node_root, node_comm);

  return new_npix;

} // compute_weights_shmem_pol_I()

template <typename dint, typename dfloat>
void compute_weights_shmem_pol_QU(           //
    const ssize_t npix,                      //
    const ssize_t nsamples,                  //
    const dint *__restrict pointings,        //
    const bool *__restrict pointings_flag,   //
    const dfloat *__restrict noise_weights,  //
    const dfloat *__restrict pol_angles,     //
    dint *__restrict node_hit_counts,        //
    MPI_Win &win_hit_counts,                 //
    dfloat *__restrict node_weighted_counts, //
    MPI_Win &win_weighted_counts,            //
    dfloat *__restrict sin2phi,              //
    dfloat *__restrict cos2phi,              //
    dfloat *__restrict node_weighted_sin_sq, //
    MPI_Win &win_weighted_sin_sq,            //
    dfloat *__restrict node_weighted_cos_sq, //
    MPI_Win &win_weighted_cos_sq,            //
    dfloat *__restrict node_weighted_sincos, //
    MPI_Win &win_weighted_sincos,            //
    dfloat *__restrict one_over_determinant, //
    const ssize_t node_root,                 //
    const bool grp_reduce,                   //
    const MPI_Comm tree_grp_comm,            //
    const MPI_Comm tree_grp_root_comm,       //
    const MPI_Comm node_comm,                //
    const MPI_Comm node_root_comm            //
) {

  int tree_grp_rank, tree_grp_size, node_rank;
  MPI_Comm_rank(tree_grp_comm, &tree_grp_rank);
  MPI_Comm_size(tree_grp_comm, &tree_grp_size);
  MPI_Comm_rank(node_comm, &node_rank);

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < nsamples; ++idx) {
    dfloat angle = pol_angles[idx];
    sin2phi[idx] = std::sin(2.0 * angle);
    cos2phi[idx] = std::cos(2.0 * angle);
  } // for

  dint *grp_hit_counts = nullptr;
  dfloat *grp_weighted_counts = nullptr;
  dfloat *grp_weighted_sin_sq = nullptr;
  dfloat *grp_weighted_cos_sq = nullptr;
  dfloat *grp_weighted_sincos = nullptr;
  SharedMemoryAllocator *allocator = nullptr;

  if (tree_grp_size == 1) {
    grp_hit_counts = node_hit_counts;
    grp_weighted_counts = node_weighted_counts;
    grp_weighted_sin_sq = node_weighted_sin_sq;
    grp_weighted_cos_sq = node_weighted_cos_sq;
    grp_weighted_sincos = node_weighted_sincos;
  } else {
    allocator = new SharedMemoryAllocator(tree_grp_comm, 0);
    grp_hit_counts = allocator->allocate<dint>(npix);
    grp_weighted_counts = allocator->allocate<dfloat>(npix);
    grp_weighted_sin_sq = allocator->allocate<dfloat>(npix);
    grp_weighted_cos_sq = allocator->allocate<dfloat>(npix);
    grp_weighted_sincos = allocator->allocate<dfloat>(npix);
  } // if

  if (allocator) {
    allocator->fence(0);
  } // if

  // Accumulation over group roots
  for (ssize_t idx = 0; idx < tree_grp_size; ++idx) {
    if (tree_grp_rank == idx) {

      accumulate_weights_pol_QU<dint, dfloat>( //
          nsamples,                            //
          pointings,                           //
          pointings_flag,                      //
          noise_weights,                       //
          pol_angles,                          //
          sin2phi,                             //
          cos2phi,                             //
          grp_hit_counts,                      //
          grp_weighted_counts,                 //
          grp_weighted_sin_sq,                 //
          grp_weighted_cos_sq,                 //
          grp_weighted_sincos                  //
      );

    } // if

    if (allocator) {
      allocator->fence(0);
    } // if
  }   // for

  // Group roots to node root reduction on each node
  if (grp_reduce) {
    if (tree_grp_root_comm != MPI_COMM_NULL) {
      MPI_Reduce(grp_hit_counts, node_hit_counts, npix, mpi_get_type<dint>(),
                 MPI_SUM, 0, tree_grp_root_comm);
      MPI_Reduce(grp_weighted_counts, node_weighted_counts, npix,
                 mpi_get_type<dfloat>(), MPI_SUM, 0, tree_grp_root_comm);
      MPI_Reduce(grp_weighted_sin_sq, node_weighted_sin_sq, npix,
                 mpi_get_type<dfloat>(), MPI_SUM, 0, tree_grp_root_comm);
      MPI_Reduce(grp_weighted_cos_sq, node_weighted_cos_sq, npix,
                 mpi_get_type<dfloat>(), MPI_SUM, 0, tree_grp_root_comm);
      MPI_Reduce(grp_weighted_sincos, node_weighted_sincos, npix,
                 mpi_get_type<dfloat>(), MPI_SUM, 0, tree_grp_root_comm);
    } // if
  }   // if

  MPI_Win_fence(0, win_hit_counts);
  MPI_Win_fence(0, win_weighted_counts);
  MPI_Win_fence(0, win_weighted_sin_sq);
  MPI_Win_fence(0, win_weighted_cos_sq);
  MPI_Win_fence(0, win_weighted_sincos);

  if (allocator) {
    delete allocator;
  } // if

  // Allreduce sync across all node roots
  if (node_rank == node_root) {
    MPI_Allreduce(MPI_IN_PLACE, node_hit_counts, npix, mpi_get_type<dint>(),
                  MPI_SUM, node_root_comm);
    MPI_Allreduce(MPI_IN_PLACE, node_weighted_counts, npix,
                  mpi_get_type<dfloat>(), MPI_SUM, node_root_comm);
    MPI_Allreduce(MPI_IN_PLACE, node_weighted_sin_sq, npix,
                  mpi_get_type<dfloat>(), MPI_SUM, node_root_comm);
    MPI_Allreduce(MPI_IN_PLACE, node_weighted_cos_sq, npix,
                  mpi_get_type<dfloat>(), MPI_SUM, node_root_comm);
    MPI_Allreduce(MPI_IN_PLACE, node_weighted_sincos, npix,
                  mpi_get_type<dfloat>(), MPI_SUM, node_root_comm);
  } // if

  MPI_Win_fence(0, win_hit_counts);
  MPI_Win_fence(0, win_weighted_counts);
  MPI_Win_fence(0, win_weighted_sin_sq);
  MPI_Win_fence(0, win_weighted_cos_sq);
  MPI_Win_fence(0, win_weighted_sincos);

  if (node_rank == node_root) {
    compute_determinants_pol_QU<dint, dfloat>( //
        npix,                                  //
        node_weighted_sin_sq,                  //
        node_weighted_cos_sq,                  //
        node_weighted_sincos,                  //
        one_over_determinant                   //
    );
  } // if

  return;

} // compute_weights_shmem_pol_QU()

template <typename dint, typename dfloat>
void compute_weights_shmem_pol_IQU(          //
    const ssize_t npix,                      //
    const ssize_t nsamples,                  //
    const dint *__restrict pointings,        //
    const bool *__restrict pointings_flag,   //
    const dfloat *__restrict noise_weights,  //
    const dfloat *__restrict pol_angles,     //
    dint *__restrict node_hit_counts,        //
    MPI_Win &win_hit_counts,                 //
    dfloat *__restrict node_weighted_counts, //
    MPI_Win &win_weighted_counts,            //
    dfloat *__restrict sin2phi,              //
    dfloat *__restrict cos2phi,              //
    dfloat *__restrict node_weighted_sin_sq, //
    MPI_Win &win_weighted_sin_sq,            //
    dfloat *__restrict node_weighted_cos_sq, //
    MPI_Win &win_weighted_cos_sq,            //
    dfloat *__restrict node_weighted_sincos, //
    MPI_Win &win_weighted_sincos,            //
    dfloat *__restrict node_weighted_sin,    //
    MPI_Win &win_weighted_sin,               //
    dfloat *__restrict node_weighted_cos,    //
    MPI_Win &win_weighted_cos,               //
    dfloat *__restrict one_over_determinant, //
    const ssize_t node_root,                 //
    const bool grp_reduce,                   //
    const MPI_Comm tree_grp_comm,            //
    const MPI_Comm tree_grp_root_comm,       //
    const MPI_Comm node_comm,                //
    const MPI_Comm node_root_comm            //
) {

  int tree_grp_rank, tree_grp_size, node_rank;
  MPI_Comm_rank(tree_grp_comm, &tree_grp_rank);
  MPI_Comm_size(tree_grp_comm, &tree_grp_size);
  MPI_Comm_rank(node_comm, &node_rank);

#pragma omp parallel for simd
  for (ssize_t idx = 0; idx < nsamples; ++idx) {
    dfloat angle = pol_angles[idx];
    sin2phi[idx] = std::sin(2.0 * angle);
    cos2phi[idx] = std::cos(2.0 * angle);
  } // for

  dint *grp_hit_counts = nullptr;
  dfloat *grp_weighted_counts = nullptr;
  dfloat *grp_weighted_sin_sq = nullptr;
  dfloat *grp_weighted_cos_sq = nullptr;
  dfloat *grp_weighted_sincos = nullptr;
  dfloat *grp_weighted_sin = nullptr;
  dfloat *grp_weighted_cos = nullptr;
  SharedMemoryAllocator *allocator = nullptr;

  if (tree_grp_size == 1) {
    grp_hit_counts = node_hit_counts;
    grp_weighted_counts = node_weighted_counts;
    grp_weighted_sin_sq = node_weighted_sin_sq;
    grp_weighted_cos_sq = node_weighted_cos_sq;
    grp_weighted_sincos = node_weighted_sincos;
    grp_weighted_sin = node_weighted_sin;
    grp_weighted_cos = node_weighted_cos;
  } else {
    allocator = new SharedMemoryAllocator(tree_grp_comm, 0);
    grp_hit_counts = allocator->allocate<dint>(npix);
    grp_weighted_counts = allocator->allocate<dfloat>(npix);
    grp_weighted_sin_sq = allocator->allocate<dfloat>(npix);
    grp_weighted_cos_sq = allocator->allocate<dfloat>(npix);
    grp_weighted_sincos = allocator->allocate<dfloat>(npix);
    grp_weighted_sin = allocator->allocate<dfloat>(npix);
    grp_weighted_cos = allocator->allocate<dfloat>(npix);
  } // if

  if (allocator) {
    allocator->fence(0);
  } // if

  // Accumulation over group roots
  for (ssize_t idx = 0; idx < tree_grp_size; ++idx) {
    if (tree_grp_rank == idx) {

      accumulate_weights_pol_IQU<dint, dfloat>( //
          nsamples,                             //
          pointings,                            //
          pointings_flag,                       //
          noise_weights,                        //
          pol_angles,                           //
          sin2phi,                              //
          cos2phi,                              //
          grp_hit_counts,                       //
          grp_weighted_counts,                  //
          grp_weighted_sin_sq,                  //
          grp_weighted_cos_sq,                  //
          grp_weighted_sincos,                  //
          grp_weighted_sin,                     //
          grp_weighted_cos                      //
      );

    } // if

    if (allocator) {
      allocator->fence(0);
    } // if
  }   // for

  // Group roots to node root reduction on each node
  if (grp_reduce) {
    if (tree_grp_root_comm != MPI_COMM_NULL) {
      MPI_Reduce(grp_hit_counts, node_hit_counts, npix, mpi_get_type<dint>(),
                 MPI_SUM, 0, tree_grp_root_comm);
      MPI_Reduce(grp_weighted_counts, node_weighted_counts, npix,
                 mpi_get_type<dfloat>(), MPI_SUM, 0, tree_grp_root_comm);
      MPI_Reduce(grp_weighted_sin, node_weighted_sin, npix,
                 mpi_get_type<dfloat>(), MPI_SUM, 0, tree_grp_root_comm);
      MPI_Reduce(grp_weighted_cos, node_weighted_cos, npix,
                 mpi_get_type<dfloat>(), MPI_SUM, 0, tree_grp_root_comm);
      MPI_Reduce(grp_weighted_sin_sq, node_weighted_sin_sq, npix,
                 mpi_get_type<dfloat>(), MPI_SUM, 0, tree_grp_root_comm);
      MPI_Reduce(grp_weighted_cos_sq, node_weighted_cos_sq, npix,
                 mpi_get_type<dfloat>(), MPI_SUM, 0, tree_grp_root_comm);
      MPI_Reduce(grp_weighted_sincos, node_weighted_sincos, npix,
                 mpi_get_type<dfloat>(), MPI_SUM, 0, tree_grp_root_comm);
    } // if
  }   // if

  MPI_Win_fence(0, win_hit_counts);
  MPI_Win_fence(0, win_weighted_counts);
  MPI_Win_fence(0, win_weighted_sin_sq);
  MPI_Win_fence(0, win_weighted_cos_sq);
  MPI_Win_fence(0, win_weighted_sincos);
  MPI_Win_fence(0, win_weighted_sin);
  MPI_Win_fence(0, win_weighted_cos);

  if (allocator) {
    delete allocator;
  } // if

  // Allreduce sync across all node roots
  if (node_rank == node_root) {
    MPI_Allreduce(MPI_IN_PLACE, node_hit_counts, npix, mpi_get_type<dint>(),
                  MPI_SUM, node_root_comm);
    MPI_Allreduce(MPI_IN_PLACE, node_weighted_counts, npix,
                  mpi_get_type<dfloat>(), MPI_SUM, node_root_comm);
    MPI_Allreduce(MPI_IN_PLACE, node_weighted_sin, npix, mpi_get_type<dfloat>(),
                  MPI_SUM, node_root_comm);
    MPI_Allreduce(MPI_IN_PLACE, node_weighted_cos, npix, mpi_get_type<dfloat>(),
                  MPI_SUM, node_root_comm);
    MPI_Allreduce(MPI_IN_PLACE, node_weighted_sin_sq, npix,
                  mpi_get_type<dfloat>(), MPI_SUM, node_root_comm);
    MPI_Allreduce(MPI_IN_PLACE, node_weighted_cos_sq, npix,
                  mpi_get_type<dfloat>(), MPI_SUM, node_root_comm);
    MPI_Allreduce(MPI_IN_PLACE, node_weighted_sincos, npix,
                  mpi_get_type<dfloat>(), MPI_SUM, node_root_comm);
  } // if

  MPI_Win_fence(0, win_hit_counts);
  MPI_Win_fence(0, win_weighted_counts);
  MPI_Win_fence(0, win_weighted_sin_sq);
  MPI_Win_fence(0, win_weighted_cos_sq);
  MPI_Win_fence(0, win_weighted_sincos);
  MPI_Win_fence(0, win_weighted_sin);
  MPI_Win_fence(0, win_weighted_cos);

  if (node_rank == node_root) {
    compute_determinants_pol_IQU<dint, dfloat>( //
        npix,                                   //
        node_weighted_counts,                   //
        node_weighted_sin_sq,                   //
        node_weighted_cos_sq,                   //
        node_weighted_sincos,                   //
        node_weighted_sin,                      //
        node_weighted_cos,                      //
        one_over_determinant                    //
    );
  } // if

  return;

} // compute_weights_shmem_pol_IQU()

/////////////////////////////////////
// nanobind registration functions //
/////////////////////////////////////

template <typename dint, typename dfloat, typename device> //
void register_compute_weights(nb::module_ &m) {
  using arr_dint = nb::ndarray<dint, nb::ndim<1>, device, nb::c_contig>;
  using arr_dfloat = nb::ndarray<dfloat, nb::ndim<1>, device, nb::c_contig>;
  using arr_bool = nb::ndarray<bool, nb::ndim<1>, device, nb::c_contig>;

  auto get_comm = [](const nb::object &mpi4py_comm) -> MPI_Comm {
    return (reinterpret_cast<const PyMPICommObject *>(mpi4py_comm.ptr()))
        ->ob_mpi;
  };

  auto get_win = [](const nb::object &mpi4py_win) -> MPI_Win {
    return (reinterpret_cast<const PyMPIWinObject *>(mpi4py_win.ptr()))->ob_mpi;
  };

  m.def(
      "compute_weights_shmem_pol_I",            //
      [get_comm, get_win](                      //
          const ssize_t npix,                   //
          const ssize_t nsamples,               //
          const arr_dint pointings,             //
          const arr_bool pointings_flag,        //
          const arr_dfloat noise_weights,       //
          arr_dint node_hit_counts,             //
          const nb::object win_hit_counts,      //
          arr_dfloat node_weighted_counts,      //
          const nb::object win_weighted_counts, //
          arr_dint observed_pixels,             //
          arr_dint __old2new_pixel,             //
          arr_bool pixel_flag,                  //
          const ssize_t node_root,              //
          const bool grp_reduce,                //
          const nb::object tree_grp_comm,       //
          const nb::object tree_grp_root_comm,  //
          const nb::object node_comm,           //
          const nb::object node_root_comm       //
          ) -> dint {
        MPI_Win win_hc = get_win(win_hit_counts);
        MPI_Win win_wc = get_win(win_weighted_counts);
        return compute_weights_shmem_pol_I( //
            npix,                           //
            nsamples,                       //
            pointings.data(),               //
            pointings_flag.data(),          //
            noise_weights.data(),           //
            node_hit_counts.data(),         //
            win_hc,                         //
            node_weighted_counts.data(),    //
            win_wc,                         //
            observed_pixels.data(),         //
            __old2new_pixel.data(),         //
            pixel_flag.data(),              //
            node_root,                      //
            grp_reduce,                     //
            get_comm(tree_grp_comm),        //
            get_comm(tree_grp_root_comm),   //
            get_comm(node_comm),            //
            get_comm(node_root_comm)        //
        );
      },
      nb::arg("npix"),                             //
      nb::arg("nsamples"),                         //
      nb::arg("pointings").noconvert(),            //
      nb::arg("pointings_flag").noconvert(),       //
      nb::arg("noise_weights").noconvert(),        //
      nb::arg("node_hit_counts").noconvert(),      //
      nb::arg("win_hit_counts").noconvert(),       //
      nb::arg("node_weighted_counts").noconvert(), //
      nb::arg("win_weighted_counts").noconvert(),  //
      nb::arg("observed_pixels").noconvert(),      //
      nb::arg("__old2new_pixel").noconvert(),      //
      nb::arg("pixel_flag").noconvert(),           //
      nb::arg("node_root"),                        //
      nb::arg("grp_reduce"),                       //
      nb::arg("tree_grp_comm").noconvert(),        //
      nb::arg("tree_grp_root_comm").noconvert(),   //
      nb::arg("node_comm").noconvert(),            //
      nb::arg("node_root_comm").noconvert()        //
  );

  m.def(
      "compute_weights_shmem_pol_QU",           //
      [get_comm, get_win](                      //
          const ssize_t npix,                   //
          const ssize_t nsamples,               //
          const arr_dint pointings,             //
          const arr_bool pointings_flag,        //
          const arr_dfloat noise_weights,       //
          const arr_dfloat pol_angles,          //
          arr_dint node_hit_counts,             //
          const nb::object win_hit_counts,      //
          arr_dfloat node_weighted_counts,      //
          const nb::object win_weighted_counts, //
          arr_dfloat sin2phi,                   //
          arr_dfloat cos2phi,                   //
          arr_dfloat node_weighted_sin_sq,      //
          const nb::object win_weighted_sin_sq, //
          arr_dfloat node_weighted_cos_sq,      //
          const nb::object win_weighted_cos_sq, //
          arr_dfloat node_weighted_sincos,      //
          const nb::object win_weighted_sincos, //
          arr_dfloat one_over_determinant,      //
          const ssize_t node_root,              //
          const bool grp_reduce,                //
          const nb::object tree_grp_comm,       //
          const nb::object tree_grp_root_comm,  //
          const nb::object node_comm,           //
          const nb::object node_root_comm       //
      ) {
        MPI_Win win_hc = get_win(win_hit_counts);
        MPI_Win win_wc = get_win(win_weighted_counts);
        MPI_Win win_wss = get_win(win_weighted_sin_sq);
        MPI_Win win_wcs = get_win(win_weighted_cos_sq);
        MPI_Win win_wsc = get_win(win_weighted_sincos);
        compute_weights_shmem_pol_QU(     //
            npix,                         //
            nsamples,                     //
            pointings.data(),             //
            pointings_flag.data(),        //
            noise_weights.data(),         //
            pol_angles.data(),            //
            node_hit_counts.data(),       //
            win_hc,                       //
            node_weighted_counts.data(),  //
            win_wc,                       //
            sin2phi.data(),               //
            cos2phi.data(),               //
            node_weighted_sin_sq.data(),  //
            win_wss,                      //
            node_weighted_cos_sq.data(),  //
            win_wcs,                      //
            node_weighted_sincos.data(),  //
            win_wsc,                      //
            one_over_determinant.data(),  //
            node_root,                    //
            grp_reduce,                   //
            get_comm(tree_grp_comm),      //
            get_comm(tree_grp_root_comm), //
            get_comm(node_comm),          //
            get_comm(node_root_comm)      //
        );
      },
      nb::arg("npix"),                             //
      nb::arg("nsamples"),                         //
      nb::arg("pointings").noconvert(),            //
      nb::arg("pointings_flag").noconvert(),       //
      nb::arg("noise_weights").noconvert(),        //
      nb::arg("pol_angles").noconvert(),           //
      nb::arg("node_hit_counts").noconvert(),      //
      nb::arg("win_hit_counts").noconvert(),       //
      nb::arg("node_weighted_counts").noconvert(), //
      nb::arg("win_weighted_counts").noconvert(),  //
      nb::arg("sin2phi").noconvert(),              //
      nb::arg("cos2phi").noconvert(),              //
      nb::arg("node_weighted_sin_sq").noconvert(), //
      nb::arg("win_weighted_sin_sq").noconvert(),  //
      nb::arg("node_weighted_cos_sq").noconvert(), //
      nb::arg("win_weighted_cos_sq").noconvert(),  //
      nb::arg("node_weighted_sincos").noconvert(), //
      nb::arg("win_weighted_sincos").noconvert(),  //
      nb::arg("one_over_determinant").noconvert(), //
      nb::arg("node_root"),                        //
      nb::arg("grp_reduce"),                       //
      nb::arg("tree_grp_comm").noconvert(),        //
      nb::arg("tree_grp_root_comm").noconvert(),   //
      nb::arg("node_comm").noconvert(),            //
      nb::arg("node_root_comm").noconvert()        //
  );

  m.def(
      "compute_weights_shmem_pol_IQU",          //
      [get_comm, get_win](                      //
          const ssize_t npix,                   //
          const ssize_t nsamples,               //
          const arr_dint pointings,             //
          const arr_bool pointings_flag,        //
          const arr_dfloat noise_weights,       //
          const arr_dfloat pol_angles,          //
          arr_dint node_hit_counts,             //
          const nb::object win_hit_counts,      //
          arr_dfloat node_weighted_counts,      //
          const nb::object win_weighted_counts, //
          arr_dfloat sin2phi,                   //
          arr_dfloat cos2phi,                   //
          arr_dfloat node_weighted_sin_sq,      //
          const nb::object win_weighted_sin_sq, //
          arr_dfloat node_weighted_cos_sq,      //
          const nb::object win_weighted_cos_sq, //
          arr_dfloat node_weighted_sincos,      //
          const nb::object win_weighted_sincos, //
          arr_dfloat node_weighted_sin,         //
          const nb::object win_weighted_sin,    //
          arr_dfloat node_weighted_cos,         //
          const nb::object win_weighted_cos,    //
          arr_dfloat one_over_determinant,      //
          const ssize_t node_root,              //
          const bool grp_reduce,                //
          const nb::object tree_grp_comm,       //
          const nb::object tree_grp_root_comm,  //
          const nb::object node_comm,           //
          const nb::object node_root_comm       //
      ) {
        MPI_Win win_hc = get_win(win_hit_counts);
        MPI_Win win_wc = get_win(win_weighted_counts);
        MPI_Win win_wss = get_win(win_weighted_sin_sq);
        MPI_Win win_wcs = get_win(win_weighted_cos_sq);
        MPI_Win win_wsc = get_win(win_weighted_sincos);
        MPI_Win win_ws = get_win(win_weighted_sin);
        MPI_Win win_wc_p = get_win(win_weighted_cos);
        compute_weights_shmem_pol_IQU(    //
            npix,                         //
            nsamples,                     //
            pointings.data(),             //
            pointings_flag.data(),        //
            noise_weights.data(),         //
            pol_angles.data(),            //
            node_hit_counts.data(),       //
            win_hc,                       //
            node_weighted_counts.data(),  //
            win_wc,                       //
            sin2phi.data(),               //
            cos2phi.data(),               //
            node_weighted_sin_sq.data(),  //
            win_wss,                      //
            node_weighted_cos_sq.data(),  //
            win_wcs,                      //
            node_weighted_sincos.data(),  //
            win_wsc,                      //
            node_weighted_sin.data(),     //
            win_ws,                       //
            node_weighted_cos.data(),     //
            win_wc_p,                     //
            one_over_determinant.data(),  //
            node_root,                    //
            grp_reduce,                   //
            get_comm(tree_grp_comm),      //
            get_comm(tree_grp_root_comm), //
            get_comm(node_comm),          //
            get_comm(node_root_comm)      //
        );
      },
      nb::arg("npix"),                             //
      nb::arg("nsamples"),                         //
      nb::arg("pointings").noconvert(),            //
      nb::arg("pointings_flag").noconvert(),       //
      nb::arg("noise_weights").noconvert(),        //
      nb::arg("pol_angles").noconvert(),           //
      nb::arg("node_hit_counts").noconvert(),      //
      nb::arg("win_hit_counts").noconvert(),       //
      nb::arg("node_weighted_counts").noconvert(), //
      nb::arg("win_weighted_counts").noconvert(),  //
      nb::arg("sin2phi").noconvert(),              //
      nb::arg("cos2phi").noconvert(),              //
      nb::arg("node_weighted_sin_sq").noconvert(), //
      nb::arg("win_weighted_sin_sq").noconvert(),  //
      nb::arg("node_weighted_cos_sq").noconvert(), //
      nb::arg("win_weighted_cos_sq").noconvert(),  //
      nb::arg("node_weighted_sincos").noconvert(), //
      nb::arg("win_weighted_sincos").noconvert(),  //
      nb::arg("node_weighted_sin").noconvert(),    //
      nb::arg("win_weighted_sin").noconvert(),     //
      nb::arg("node_weighted_cos").noconvert(),    //
      nb::arg("win_weighted_cos").noconvert(),     //
      nb::arg("one_over_determinant").noconvert(), //
      nb::arg("node_root"),                        //
      nb::arg("grp_reduce"),                       //
      nb::arg("tree_grp_comm").noconvert(),        //
      nb::arg("tree_grp_root_comm").noconvert(),   //
      nb::arg("node_comm").noconvert(),            //
      nb::arg("node_root_comm").noconvert()        //
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

///////////////////////
// Module definition //
///////////////////////

NB_MODULE(compute_weights_shared, m) {
  m.doc() = "compute_weights_shared";

  register_compute_weights<int64_t, double, nb::device::cpu>(m);
  register_compute_weights<int32_t, double, nb::device::cpu>(m);
  register_compute_weights<int64_t, float, nb::device::cpu>(m);
  register_compute_weights<int32_t, float, nb::device::cpu>(m);
}
