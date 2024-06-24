# BrahMap quick start guide

Proceed with following steps:

1. Process the pointings, provided as pixel indices

    ```py
    processed_pointings = ProcessTimeSamples(
        npix=npix, 
        pointings=pointings, 
        solver_type=3, 
        pol_angles=pol_angles
    )
    npix_new, __ = processed_pointings.get_new_pixel
    ```

2. Create pointing matrix operator as sparse linear operator

    ```py
    P = SparseLO(n=npix_new, 
        m=nsamp, 
        pix_samples=processed_pointings.pixs, 
        pol=3,
        angle_processed=processed_pointings
    )
    ```

3. Make a noise covariance matrix

    ```py
    inv_N = BlockLO(blocksize=blocksize, t=inv_sigma2, offdiag=False)
    ```

4. Make a block-diagonal preconditioner operator

    ```py
    Mbd = BlockDiagonalPreconditionerLO(CES=processed_pointings,
        n=npix_new, 
        pol=3
    )
    ```

5. Solve for sky maps

    ```py
    A = P.T * inv_N * P
    b = P.T * inv_N * tod_array
    map_out = scipy.sparse.linalg.cg(A, b, M=Mbd)
    ```
