# Math Functions

## Linear algebra

- [`parallel_norm`](parallel_norm.md)
- [`cg`](cg.md)

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

- `brahmap.math.sin`: sine function
- `brahmap.math.cos`: cosine function
- `brahmap.math.tan`: tangent function
- `brahmap.math.asin`: arcsine function
- `brahmap.math.acos`: arccosine function
- `brahmap.math.atan`: arctangent function
- `brahmap.math.exp`: exponential function, $e^x$
- `brahmap.math.exp2`: exponential function with base 2, $2^x$
- `brahmap.math.log`: natural logarithm function
- `brahmap.math.log2`: base-2 logarithm function
- `brahmap.math.sqrt`: square-root function
- `brahmap.math.cbrt`: cube-root function

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
