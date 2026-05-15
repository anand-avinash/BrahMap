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

| Option          | Values               | Description                                               |
| :-------------- | :------------------- | :-------------------------------------------------------- |
| `--nside`       | e.g. `128`           | Sets $N_{side}$ (overrides npix)                        |
| `--nsamples`    | e.g. `1000000`       | Sets global number of samples (overrides global nsamples) |
| `--dtype-float` | `float32`, `float64` | Sets floating point precision (default: float64)          |
| `--dtype-int`   | `int32`, `int64`     | Sets integer precision (default: int64)                   |

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

```python
@pytest.mark.benchmark(group="extensions::my_group")
class TestMyComponent:
    def test_bench_my_function_1(self, benchmark, data):
        # Setup randomized buffers using the module-level data fixture
        rng = data["rng"]
        vec = rng.random(data["npix"]).astype(data["dtype_float"])
        
        # Run benchmark
        benchmark(my_function1, vec)

    def test_bench_my_function_2(self, benchmark, data):
        # Setup randomized buffers using the module-level data fixture
        rng = data["rng"]
        vec = rng.random(data["npix"]).astype(data["dtype_float"])
        
        # Run benchmark
        benchmark(my_function2, vec)
```
