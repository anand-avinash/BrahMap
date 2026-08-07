# Math Functions

<!-- markdownlint-disable MD013 -->

## Linear algebra

| Function | Description |
| :--- | :--- |
| [`parallel_norm`](parallel_norm.md) | Computes the $L_2$ norm of a distributed vector across MPI processes. |
| [`cg`](cg.md) | Solves the linear system $Ax = b$ using the Conjugate Gradient (CG) method. |

<!-- markdownlint-enable MD013 -->

## Unary functions

These functions are element-wise unary functions, and have common signature:

```python
function_name(
    size,     # size of the input/output array
    vec,      # input array
    result,   # output array containing the result (overwritten)
)
```

Following functions are available:

| Function | Description |
| :--- | :--- |
| `brahmap.math.sin` | Element-wise sine of an array. |
| `brahmap.math.cos` | Element-wise cosine of an array. |
| `brahmap.math.tan` | Element-wise tangent of an array. |
| `brahmap.math.asin` | Element-wise arcsine of an array. |
| `brahmap.math.acos` | Element-wise arccosine of an array. |
| `brahmap.math.atan` | Element-wise arctangent of an array. |
| `brahmap.math.exp` | Element-wise exponential ($e^x$) of an array. |
| `brahmap.math.exp2` | Element-wise base-2 exponential ($2^x$) of an array. |
| `brahmap.math.log` | Element-wise natural logarithm of an array. |
| `brahmap.math.log2` | Element-wise base-2 logarithm of an array. |
| `brahmap.math.sqrt` | Element-wise square root of an array. |
| `brahmap.math.cbrt` | Element-wise cube root of an array. |

## `dtype` hints

- **`DTypeFloat`**

    ::: brahmap.math.DTypeFloat
        options:
          show_root_heading: false

- **`DTypeInt`**

    ::: brahmap.math.DTypeInt
        options:
          show_root_heading: false

- **`DTypeUInit`**

    ::: brahmap.math.DTypeUInit
        options:
          show_root_heading: false

- **`DTypeBool`**

    ::: brahmap.math.DTypeBool
        options:
          show_root_heading: false
