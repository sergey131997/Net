import math


class Error(object):
    """docstring for Error"""
    def GetError(func, x):
        if func == "sqr":
            return x**2
        if func == "atan":
            return math.atan(x)
        if func == "relu":
            return max(0, x)
        if func == "rf":
            return x / ((abs(x) + 1) ** 2)

    def GetGrad(func, x):
        if func == "sqr":
            return (2 * x)
        if func == "atan":
            return 1 / (x**2 + 1)
        if func == "relu":
            return 1 if x > 0 else 0
        if func == "rf":
            return ((abs(x) + 1) - x * (abs(x))) / ((abs(x) + 1) ** 2)
