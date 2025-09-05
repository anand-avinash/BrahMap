#!/bin/bash

# This script simply executes the example notebooks and test scripts
ipynb_filename=("rectangular_I_map_explicit.ipynb"
    "healpix_QU_map_wrapper.ipynb"
    "lbsim_IQU_map_explicit.ipynb"
    "lbsim_IQU_map_wrapper.ipynb"
    "basic_noise_covariances.ipynb"
    "block_diagonal_noise_covariances.ipynb"
    "lbsim_noise_covariances.ipynb"
)

py_filename=("rectangular_I_map_wrapper.py"
    "healpix_QU_map_explicit.py"
    "lbsim_IQU_map_wrapper.py"
)

### Loop over the jupyter notebooks
for filename in "${ipynb_filename[@]}"; do
    printf "\n\nExecuting ${filename}...\n"
    jupyter nbconvert --to notebook --execute ${filename} --output notebook_test.ipynb
    printf "Execution successful\n\n"
done

for filename in "${py_filename[@]}"; do
    for nprocs in 1 2 5; do
        printf "\n\nExecuting ${filename} with ${nprocs} nprocs...\n"
        mpiexec --map-by :OVERSUBSCRIBE -n ${nprocs} python ${filename}
        printf "Execution successful\n\n"
    done
done
