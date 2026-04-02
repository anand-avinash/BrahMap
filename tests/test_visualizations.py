import pytest
import unittest.mock as mock
import numpy as np

# Skip the whole module if matplotlib is not installed.
pytest.importorskip("matplotlib")

from brahmap.utilities.visualizations import plot_LinearOperator

class DummyLinearOperator:
    def __init__(self, nargin, nargout, row_size=None, col_size=None):
        self.nargin = nargin
        self.nargout = nargout
        if row_size is not None:
            self.row_size = row_size
        if col_size is not None:
            self.col_size = col_size

    def to_array(self):
        return np.ones((self.nargout, self.nargin))

@mock.patch("brahmap.utilities.visualizations.plt")
def test_plot_LinearOperator_no_sizes(mock_plt):
    """Test plot_LinearOperator when operator does not have row_size/col_size attrs."""
    nargin = 3
    nargout = 4
    operator = DummyLinearOperator(nargin=nargin, nargout=nargout)

    # Mock plt.gca() to return a mock axes
    mock_ax = mock.MagicMock()
    mock_plt.gca.return_value = mock_ax

    plot_LinearOperator(operator)

    mock_plt.figure.assert_called_once()
    mock_plt.imshow.assert_called_once()

    # Ensure to_array result was passed to imshow
    args, kwargs = mock_plt.imshow.call_args
    assert np.array_equal(args[0], np.ones((nargout, nargin)))

    mock_plt.tick_params.assert_called_once_with(
        axis="both", bottom=False, left=False, labelbottom=False, labelleft=False
    )

    # Since row_size/col_size aren't set, they default to [nargout] and [nargin] respectively
    # tic is -0.5, len(col_size) is 1, so xticks should be [tic + 0] = [-0.5]
    mock_ax.set_xticks.assert_called_once_with([-0.5], minor=True)
    # len(row_size) is 1, so yticks should be [tic + 0] = [-0.5]
    mock_ax.set_yticks.assert_called_once_with([-0.5], minor=True)

    mock_ax.grid.assert_called_once_with(which="minor", color="w", linestyle="-", linewidth=1)
    mock_ax.tick_params.assert_called_once_with(which="minor", bottom=False, left=False)

    mock_plt.colorbar.assert_called_once()

@mock.patch("brahmap.utilities.visualizations.plt")
def test_plot_LinearOperator_with_sizes(mock_plt):
    """Test plot_LinearOperator when operator has row_size/col_size attrs."""
    nargin = 5
    nargout = 6
    operator = DummyLinearOperator(
        nargin=nargin, nargout=nargout, row_size=[2, 4], col_size=[1, 4]
    )

    # Mock plt.gca() to return a mock axes
    mock_ax = mock.MagicMock()
    mock_plt.gca.return_value = mock_ax

    plot_LinearOperator(operator)

    mock_plt.figure.assert_called_once()
    mock_plt.imshow.assert_called_once()

    # Check ticks with row_size and col_size
    # col_size=[1, 4] -> indices 0, 1 -> sums: 0, 1 -> xticks: [-0.5, 0.5]
    # row_size=[2, 4] -> indices 0, 1 -> sums: 0, 2 -> yticks: [-0.5, 1.5]
    mock_ax.set_xticks.assert_called_once_with([-0.5, 0.5], minor=True)
    mock_ax.set_yticks.assert_called_once_with([-0.5, 1.5], minor=True)
