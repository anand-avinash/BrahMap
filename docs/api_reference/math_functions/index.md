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

- `DTypeFloat`: type-hint for the `dtype` of floating-point numbers
- `DTypeInt`: type-hint for the `dtype` of signed integers
- `DTypeUInit`: type-hint for the `dtype` of unsigned integers
- `DTypeBool`: type-hint for the `dtype` of bools
