<!-- markdownlint-disable MD013 -->

# Benchmarking BrahMap

This directory contains scripts for benchmarking the performance of BrahMap components. While the benchmark scripts are primarily designed for serial execution, they can also be used for parallel execution with `mpirun/mpiexec`, though one has to be careful while saving the benchmark results.

## Setup

The benchmarks use [`pytest-benchmark`](http://pytest-benchmark.readthedocs.org/en/stable/). You can install it via pip:

```bash
pip install pytest-benchmark
```

To produce the benchmark histograms as well, install `pygal` and `pygaljs` packages with `pip install pygal pygaljs`.

## Running Benchmarks

The benchmarks can be run with `pytest`:

```bash
# To run all the benchmarks
pytest benchmarks/

# To run the specific benchmarks
pytest benchmarks/test_bench_extensions.py

# To run the specific benchmarks (e.g., TestMath class)
pytest benchmarks/test_bench_extensions.py::TestMath

# It is often useful to set explicitly the number of parallel OpenMP threads
OMP_NUM_THREADS=1 pytest benchmarks/

# To generate histograms (SVG files with performance statistics)
pytest benchmarks/ --benchmark-histogram
```

### Customizing Benchmark Size

You can control the size of the data used in benchmarks using the `--size` flag:

```bash
pytest benchmarks/ --size=small
```

Available sizes are:

| Size     | npix ($N_{side}$)         | Global nsamples |
| :------- | :------------------------ | :-------------- |
| `small`  | $12 \times 256^2$ (256)   | $10^5$          |
| `medium` | $12 \times 512^2$ (512)   | $10^6$          |
| `large`  | $12 \times 1024^2$ (1024) | $10^8$          |

This allows for quick sanity checks with `small` or more robust performance measurements with `large`. The samples are automatically distributed across available MPI ranks.

### Overriding Benchmark Parameters

You can override specific parameters regardless of the `--size` flag:

| Option             | Values               | Description                                               |
| :----------------- | :------------------- | :-------------------------------------------------------- |
| `--nside`          | e.g. `128`           | Sets $N_{side}$ (overrides npix)                        |
| `--nsamples`       | e.g. `1000000`       | Sets global number of samples (overrides global nsamples) |
| `--dtype-float`    | `float32`, `float64` | Sets floating point precision (default: float64)          |
| `--dtype-int`      | `int32`, `int64`     | Sets integer precision (default: int64)                   |
| `--mpi-rounds`        | e.g. `20`            | Sets number of rounds for `mpi_benchmark` (default: 20)      |
| `--mpi-iterations`    | e.g. `1`             | Sets iterations per round for `mpi_benchmark` (default: 1)   |
| `--mpi-warmup-rounds` | e.g. `0`             | Sets number of warmup rounds for `mpi_benchmark` (default: 0)|

Example usage:

```bash
pytest benchmarks/ --nside=256 --dtype-float=float32
```

### Saving and Comparing Results

To save the benchmark results to a JSON file:

```bash
pytest benchmarks/ --benchmark-json results_v1.json
```

To compare two or more saved JSON results:

```bash
pytest-benchmark compare results_v1.json results_v2.json
```

This will show a detailed comparison table with percentage differences. You can also use the option `--sort=...` to sort the comparison results.

### Rounds vs. Iterations

- **Iterations (per round):** The number of times the function is called consecutively **within a single timed measurement round**. The iteration count should be set according to the function being benchmarked:
  - For fast functions (taking microseconds), a higher value (e.g., `100` or `1000`) is more suitable to amortize timing overhead.
  - For computationally heavy/slower functions (taking milliseconds/seconds), a lower value (e.g., `1` or `5`) is sufficient.
  - For MPI-related benchmarks, the default number of iterations is `1`. It can be overridden using `--mpi-iterations=<num_iterations>`.
- **Rounds:** The number of independent measurements taken to compute statistics. A round **constitutes a single timed measurement event**. This should be set to achieve a balance between total run time and the statistical stability of the measurements.
  - For MPI-related tests, the default number of rounds is `20`. It can be overridden using `--mpi-rounds=<num_rounds>`.

## Parallel Execution (with MPI)

To run benchmarks in parallel:

```bash
# Run benchmarks across multiple ranks
mpirun -n 4 pytest benchmarks/

# To save results per rank, preventing overwrites (this works for OpenMPI)
mpirun -n 4 pytest benchmarks/ --benchmark-json=results_rank_${OMPI_COMM_WORLD_RANK}.json
```

For collective operations (like `*_rmult`), it is recommended to call the individual benchmarks one-by-one to ensure ranks are synchronized for accurate measurements.

## Adding New Benchmarks

New benchmarks should follow the class-based structure as in `test_bench_extensions.py`. This groups related benchmarks in the final report.

### 1. Choosing the Right Benchmark Fixture

When writing a benchmark, request the appropriate fixture depending on the operations being measured:

- **`benchmark` (Standard):** For non-MPI / local operations (e.g. math functions, local matrix multiplications). It uses dynamic calibration to scale rounds and iterations automatically.

- **`mpi_benchmark` (Custom):** For collective MPI or shared-memory operations (e.g., projection operators, weight computation). It uses fixed numbers of rounds and iterations (default: 20 rounds of 1 iteration) to prevent rank desynchronization.

### 2. Writing the Benchmark Code

```python
# Case A: Standard local benchmark using 'benchmark'
@pytest.mark.benchmark(group="extensions::my_group")
class TestMyComponent:
    def test_bench_my_function(self, benchmark, data):
        vec = data["rng"].random(data["npix"]).astype(data["dtype_float"])
        benchmark(my_function, vec)

# Case B: MPI collective benchmark using 'mpi_benchmark' with setup reset
@pytest.mark.benchmark(group="extensions::my_group")
class TestMyMPIComponent:
    def test_bench_my_mpi_function(self, mpi_benchmark, data):
        prod = np.zeros(data["npix"])
        vec = data["rng"].random(data["npix"]).astype(data["dtype_float"])
        
        # Setup runs before each round to reset arrays and prevent numerical 
        # overflow
        def setup():
            prod.fill(0)
            return (data["npix"], vec, prod), {}
            
        mpi_benchmark(my_mpi_function, setup=setup)
```
