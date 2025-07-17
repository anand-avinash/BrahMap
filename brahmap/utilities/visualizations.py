import numpy as np
import matplotlib.pyplot as plt

from ..base import LinearOperator


def plot_LinearOperator(
    operator: LinearOperator,
):
    """A utility function to visualize BrahMap linear operators. Make sure that `matplotlib` is installed if you want to use it.

    !!! Warning

        This method first allocates a NumPy array of shape `self.shape`
        and data-type `self.dtype`, and then fills them with numbers. As
        such it can occupy an enormous amount of memory. Don't use it
        unless you understand the risk!

    Parameters
    ----------
    operator : LinearOperator
        _description_
    """
    plt.figure()
    plt.imshow(operator.to_array())

    plt.tick_params(
        axis="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    ax = plt.gca()

    row_size = getattr(operator, "row_size", [operator.nargout])
    col_size = getattr(operator, "col_size", [operator.nargin])

    tic = -0.5
    ax.set_xticks(
        [tic + np.sum(col_size[:idx]) for idx in range(len(col_size))],
        minor=True,
    )
    ax.set_yticks(
        [tic + np.sum(row_size[:idx]) for idx in range(len(row_size))],
        minor=True,
    )

    ax.grid(which="minor", color="w", linestyle="-", linewidth=1)

    ax.tick_params(which="minor", bottom=False, left=False)
    plt.colorbar()
