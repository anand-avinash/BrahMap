import warnings


class TypeChangeWarning(Warning):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class LowerTypeCastWarning(Warning):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


def filter_warnings(wfilter):
    """
    wfilter: {string}
    - "ignore": never print matching warnings;
    - "always": always print matching warnings

    """
    warnings.simplefilter(wfilter)


class ShapeError(Exception):
    """
    Exception class for handling shape mismatch errors.

    Exception raised when defining a linear operator of the wrong shape or
    multiplying a linear operator with a vector of the wrong shape.

    """

    def __init__(self, value):
        super(ShapeError, self).__init__()
        self.value = value

    def __str__(self):
        return repr(self.value)
