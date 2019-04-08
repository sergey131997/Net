import math
import random
import errors
from copy import deepcopy


class Layers(object):
    """docstring for Conv"""

    def __init__(self, str):
        super(Layers, self).__init__()
        self.weights = []
        self.address = []
        self.values = []
        self.biases = []
        stride_y = 1
        stride_x = 1
        self.channels = 0
        self.width = 0
        self.height = 0
        self.func = "relu"
        self.grad = []
        self.name = str

    def Conv(self, width, height, channels,
             stride_x, stride_y, func, layer):
        self.weights =\
            [[[0 for i in range(layer.width - width + 1) // stride_x] for j in range(
                (layer.height - height + 1) // stride_y)] for z in range(channels)]
        self.values =\
            [[[0 for i in range(layer.width - width + 1) // stride_x] for j in range(
                (layer.height - height + 1) // stride_y)] for z in range(channels)]
        self.biases =\
            [[[random.random() for i in range(layer.width - width + 1) // stride_x] for j in range(
                (layer.height - height + 1) // stride_y)] for z in range(channels)]
        self.grad =\
            [[[0 for i in range(layer.width - width + 1) // stride_x] for j in range(
                (layer.height - height + 1) // stride_y)] for z in range(channels)]
        self.address =\
            [[[0 for i in range(layer.width - width + 1) // stride_x] for j in range(
                (layer.height - height + 1) // stride_y)] for z in range(channels)]

        self.height = len(self.weights)
        self.width = len(self.weights[0])
        self.channels = channels
        self.func = func

        for c in self.channels:
            for y in self.channels:
                for x in self.width:
                    self.weights[c][y][x] = Layers.init_weight(
                        width, height, layers.channels)
                    self.address[c][y][x] = [[[[c, (x + w) * stride_x, (y + h) * stride_y] for w in range(width)]
                                              for h in range(height)] for c in range(layer.channels)]
        return self

    def Data(self, width, height, channels):
        self.values = [[[0 for i in range(width)]
                        for j in range(height)] for c in range(channels)]
        self.stride_y = 1
        self.stride_x = 1
        self.height = height
        self.width = width
        self.channels = channels
        # print(width, height, channels)
        return self

    def init_weight(width, height, channels):
        ans = []
        for c in range(channels):
            ss = 0
            tmp = [[random.random() for x in range(width)]
                   for j in range(height)]
            for y in range(height):
                for x in range(width):
                    ss += tmp[y][x]

            ans.append([[tmp[y][x] / ss for x in range(width)]
                        for j in range(height)])
        return ans

    def FullyConnected(self, width, height, channels, func, layer):
        self.weights =\
            [[[0 for i in range(width)] for j in range(height)]
             for z in range(channels)]
        self.values =\
            [[[0 for i in range(width)] for j in range(height)]
             for z in range(channels)]
        self.biases =\
            [[[random.random() for i in range(width)] for j in range(height)]
             for z in range(channels)]
        self.address =\
            [[[0 for i in range(width)] for j in range(height)]
             for z in range(channels)]
        self.grad =\
            [[[0 for i in range(width)] for j in range(height)]
             for z in range(channels)]

        self.width = width
        self.height = height
        self.channels = channels
        self.func = func
        self.stride_x = 1
        self.stride_y = 1

        for c in range(self.channels):
            for y in range(self.height):
                for x in range(self.width):
                    self.weights[c][y][x] = Layers.init_weight(
                        layer.width, layer.height, layer.channels)
                    self.address[c][y][x] = [[[[c, h, w] for w in range(layer.width)]
                                              for h in range(layer.height)] for c in range(layer.channels)]
        return self

    def MakeArray(grad):
        for c in range(len(grad)):
            if type(grad[c]) != type([]):
                grad[c] = []
            else:
                Layers.MakeArray(grad[c])

    def ApplyConv(weights, address, values):
        ss = 0
        for c in range(len(weights)):
            for h in range(len(weights[c])):
                for w in range(len(weights[c][h])):
                    c1, h1, w1 = address[c][h][w]
                    ss += weights[c][h][w] * values[c1][h1][w1]
        return ss

    def RecomputeGradient(self, grad, values, coef_w, coef_b):

        grad_w = deepcopy(self.weights)
        Layers.MakeArray(grad_w)

        grad_x = deepcopy(values)
        Layers.MakeArray(grad_x)

        grad_b = deepcopy(self.values)
        Layers.MakeArray(grad_b)

        for c in range(self.channels):
            for h in range(self.height):
                for w in range(self.width):
                    grad_b[c][h][w].append(errors.Error.GetGrad(
                        self.func,
                        Layers.ApplyConv(self.weights[c][h][w],
                                         self.address[c][h][w],
                                         values) + self.biases[c][h][w]) * grad[c][h][w])
                    Layers.ComputeGradConv(
                        grad_w[c][h][w],
                        grad_x,
                        grad_b[c][h][w][-1],
                        self.address[c][h][w],
                        values,
                        self.weights[c][h][w])

        for c in range(self.channels):
            for h in range(self.height):
                for w in range(self.width):
                    for c1 in range(len(self.weights[c][h][w])):
                        for h1 in range(len(self.weights[c][h][w][c1])):
                            for w1 in range(len(self.weights[c][h][w][c1][h1])):
                                c_t, h_t, w_t = self.address[c][h][w][c1][h1][w1]
                                summ = sum(
                                    grad_w[c][h][w][c1][h1][w1]) / len(grad_w[c][h][w][c1][h1][w1])
                                self.weights[c][h][w][c1][h1][w1] -= coef_w * summ
                    grad_b[c][h][w] = sum(
                        grad_b[c][h][w]) / len(grad_b[c][h][w])
                    self.biases[c][h][w] -= grad_b[c][h][w] * coef_b

        for c in range(len(grad_x)):
            for h in range(len(grad_x[c])):
                for w in range(len(grad_x[c][h])):
                    grad_x[c][h][w] = sum(
                        grad_x[c][h][w]) / len(grad_x[c][h][w])

        return grad_x

    def ComputeGradConv(grad, grad_x, grad_error, address, values, weights):
        for c in range(len(grad)):
            for h in range(len(grad[c])):
                for w in range(len(grad[c][h])):
                    c1, h1, w1 = address[c][h][w]
                    grad[c][h][w].append(grad_error * values[c1][h1][w1])
                    grad_x[c1][h1][w1].append(grad_error * weights[c][h][w])
