import warnings
from typing import Literal


class TypeChangeWarning(Warning):
    """A custom warning raised when a variable's data type is forced to be changed.

    Parameters
    ----------
    message : str
        The descriptive warning message
    """

    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self):
        return repr(self.message)


class LowerTypeCastWarning(Warning):
    """A custom warning raised when an array is cast to a lower precision data type.

    Parameters
    ----------
    message : str
        The descriptive warning message
    """

    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self):
        return repr(self.message)


def filter_warnings(wfilter: Literal["ignore", "always"]) -> None:
    """Configures the global Python warning filter behavior for the framework.

    Parameters
    ----------
    wfilter : Literal["ignore", "always"]
        The action to take when a warning is triggered. Use "ignore" to suppress
        warnings, or "always" to always print the warnings
    """
    warnings.simplefilter(wfilter)


class ShapeError(Exception):
    """An exception class for handling array or operator shape mismatch errors.

    This exception is raised when defining a linear operator with an invalid shape
    or when multiplying a linear operator by a vector of an incompatible shape.

    Parameters
    ----------
    value : str
        The descriptive error message detailing the shape mismatch
    """

    def __init__(self, value: str) -> None:
        super(ShapeError, self).__init__()
        self.value = value

    def __str__(self):
        return repr(self.value)
