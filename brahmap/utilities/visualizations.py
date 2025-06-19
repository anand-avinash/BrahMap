import numpy as np
import matplotlib.pyplot as plt


def plot_LinearOperator(
    operator: "LinearOperator",  # noqa
):
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
