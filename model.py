from layer import Layers
from errors import Error
import math
from matplotlib import pyplot
import random
import json


class Model(object):
    """docstring for Model"""

    def __init__(self, coef_w, coef_b, use_softmax):
        self.layers = []
        self.coef_w = coef_w
        self.coef_b = coef_b
        self.use_softmax = use_softmax

    def AddLayer(self, type_, width, height, channels, func="tanh", stride_x=0, stride_y=0):

        if type_ == "conv":
            self.layers.append(Layer.Conv(Layers(len(self.layers)), width, height,
                                          channels, stride_x, stride_y, func, self.layers[-1]))
            return
        if type_ == "full":
            self.layers.append(Layers.FullyConnected(Layers(len(self.layers)),
                                                     width, height, channels, func, self.layers[-1]))
            return

        if type_ == "data":
            self.layers.append(Layers.Data(
                Layers(len(self.layers)), width, height, channels))
            return

    def GetClass(self, y):
        i = -1
        d = 0
        idx = 0
        for y_ in self.classes:
            dis = Model.SqrErr(y_, y)
            if i == -1 or d > dis:
                d = dis
                i = idx
            idx += 1
        return i

    def AddClass(self, y):
        for i in self.classes:
            if i == y:
                return
        self.classes.append(y)

    def Train(self, x, y, epoches, train_percent, save_path, visualis=False):
        test = len(x) * train_percent
        test_set = set()
        self.classes = []
        while(len(test_set) < test):
            test_set.add(random.randint(0, len(x) - 1))
        test_x, test_y, train_x, train_y = [], [], [], []
        for i in range(len(x)):
            if i in test_set:
                test_x.append(x[i])
                test_y.append(y[i])
            else:
                train_x.append(x[i])
                train_y.append(y[i])
            self.AddClass(y[i])

        best_error = 100000
        tee = []
        tre = []
        ep = []
        if len(x[0][0][0]) == 2:
            pyplot.figure()
            for class_ in self.classes:
                j = [k for k in range(len(x)) if y[k] == class_]
                pyplot.scatter([x[jj][0][0][0]
                                for jj in j], [x[jj][0][0][1] for jj in j])
            pyplot.savefig(save_path + "_ideal.png")
            pyplot.close()
        step = len(train_x)
        self.epoch_size = 100
        step_c=0
        with open(save_path[:-5] + "_log.txt", "w") as ofs:
            for epoche in range(epoches):
                if (step + 1) * self.epoch_size >= len(train_x):
                    index = [i for i in range(len(train_x))]
                    random.shuffle(index)
                    i_x = [train_x[i] for i in index]
                    i_y = [train_y[i] for i in index]
                    step = 0

                self.Teach([i_x], [i_y], step * self.epoch_size,
                           (step + 1) * self.epoch_size)
                ep.append(epoche)
                tee.append(self.LearnError([test_x], [test_y]))
                tre.append(self.LearnError(
                    [i_x], [i_y], step * self.epoch_size, (step + 1) * self.epoch_size)),

                if best_error > tee[-1]:
                    best_error = tee[-1]
                    self.DumpModel(save_path)

                print("Epoch: {};\n{}/{}".format(epoche, tre[-1], tee[-1]))
                ofs.write(
                    "Epoch: {};\n{}/{}\n".format(epoche, tre[-1], tee[-1]))

                if len(x[0][0][0]) == 2 and visualis:
                    h = [[] for i in range(len(self.classes))]
                    for i in range(len(x)):
                        for j in range(len(x[i])):
                            class_ = self.GetClass(self.Solve([x[i][j]]))
                            h[class_].append(x[i][j][0])
                    pyplot.figure()
                    for hh in h:
                        pyplot.scatter([hh[i][0] for i in range(len(hh))],
                                       [hh[i][1] for i in range(len(hh))])
                    pyplot.savefig(save_path + "_classification.png")
                    pyplot.close()

                pyplot.figure()
                pyplot.plot(ep, tre)
                pyplot.plot(ep, tee)
                pyplot.savefig(save_path + "_error.png")
                pyplot.close()
                step_c+=1

                if len(tee) > 1:
                    if tee[-1] > tee[-2] or step_c > 30:
                        step += 1
                        step_c = 0


    def LearnError(self, x, y, start=-1, end=-1):
        squared_error = 0
        if start == -1:
            start = 0
            end = len(x[0])
        if end > len(x[0]):
            end = len(x[0])
        for i in range(start, end):
            squared_error += Model.SqrErr(self.Solve(x[0][i]), y[0][i])
        return squared_error / (end - start)

    def SqrErr(a, b):
        sqrerr = 0
        # print(a, b)
        for i in range(len(a)):
            sqrerr += (a[i] - b[i]) ** 2
        return sqrerr

    def Solve(self, x):
        for c in range(self.layers[0].channels):
            for h in range(self.layers[0].height):
                for w in range(self.layers[0].width):
                    self.layers[0].values[c][h][w] = x[c][h][w]

        for i in range(1, len(self.layers)):
            for c in range(self.layers[i].channels):
                for h in range(self.layers[i].height):
                    for w in range(self.layers[i].width):
                        self.layers[i].values[c][h][w] = Error.GetError(
                            self.layers[i].func,
                            Layers.ApplyConv(self.layers[i].weights[c][h][w],
                                             self.layers[i].address[c][h][w],
                                             self.layers[i - 1].values) + self.layers[i].biases[c][h][w])

        ans = [self.layers[-1].values[i][0][0] for i in range(self.layers[-1].channels)]

        if self.use_softmax:
            summ = sum([math.exp(ans[i]) for i in range(len(ans))])
            ans = [math.exp(ans[i]) / summ for i in range(len(ans))]

        return ans

    def Teach(self, x, y, start, end):
        if end > len(x[0]):
            end = len(x[0])
        for jj in range(start, end):
            y_ = self.Solve(x[0][jj])
            grad = [[[2 * (- y[0][jj][i] + y_[i])]]
                    for i in range(self.layers[-1].channels)]
            
            if self.use_softmax:
                for i in range(len(y_)):
                    grad[i][0][0] = grad[i][0][0] * y_[i] * (1 - y_[i])

            for i in range(len(self.layers) - 1, 0, -1):
                grad = self.layers[i].RecomputeGradient(
                    grad, self.layers[i - 1].values, self.coef_w, self.coef_b)

    def DumpModel(self, path):
        model = []
        model.append({"width": self.layers[0].width,
                      "height": self.layers[0].height,
                      "channels": self.layers[0].channels})
        for i in range(1, len(self.layers)):
            model.append({
                "width": self.layers[i].width,
                "height": self.layers[i].height,
                "channels": self.layers[i].channels,
                "func": self.layers[i].func,
                "weights": self.layers[i].weights,
                "address": self.layers[i].address,
                "biases": self.layers[i].biases})
        model.append(self.classes)
        json.dump(model, open(path, "w"), indent=2)

    def LoadModel(self, path):
        self.layers = []
        js = json.load(open(path))
        self.layers.append(Layers.Data(Layers(len(self.layers)),
                                       js[0]["width"], js[0]["height"], js[0]["channels"]))
        for i in range(1, len(js) - 1):
            layer = Layers(Layers(len(self.layers)))
            layer.width = int(js[i]["width"])
            layer.height = js[i]["height"]
            layer.channels = js[i]["channels"]
            layer.func = js[i]["func"]
            layer.weights = js[i]["weights"]
            layer.address = js[i]["address"]
            layer.biases = js[i]["biases"]
            layer.values = [[[0 for w in range(layer.width)] for h in range(
                layer.height)] for c in range(layer.channels)]
            self.layers.append(layer)
        self.classes = js[-1]
