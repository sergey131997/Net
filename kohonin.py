import random
import json
from matplotlib import pyplot
import math


class Kohonin(object):
    """docstring for Kohonin"""

    def __init__(self, input_size, output_size):
        self.map = [[[0 for i in range(input_size)] for w in range(
            output_size[0])] for h in range(output_size[1])]
        self.classes = []
        self.func = ""
        self.width, self.height = output_size
        self.v_size = input_size
        for h in range(output_size[1]):
            for w in range(output_size[0]):
                summ = 0
                for j in range(input_size):
                    s = random.random()
                    self.map[h][w][j] = s
                    summ += s**2
                self.map[h][w] = [self.map[h][w][j] /
                                  summ for j in range(input_size)]

    def TrainKohonin(self, data, save_path, epoch=10):
        for t in range(epoch):
            err = 0
            pyplot.figure()
            x_ = []
            y_ = []
            for ex in data:
                d = self.GetCloses(ex)
                err += Kohonin.Distance(ex, self.map[d[1]][d[0]])
                x_.append(d[0])
                y_.append(d[1])

            pyplot.scatter(x_, y_)
            pyplot.savefig(save_path[:-4] + "png")
            pyplot.close()
            err /= len(data)

            print("Epoch:{}\n{}".format(t, err))

            for ex in data:
                d = self.GetCloses(ex)
                self.Update(d, ex, t)

    def TrainClassifier(self, data, class_number, func, save_path, epoch):
        self.classes = []
        self.func = func

        b_error = 1000

        for i in range(epoch):
            cl, err = self.TrainClassifierIteration(data, class_number, func)
            print ("Epoch: {}\n{}".format(i, err))
            if err < b_error:
                b_error = err
                self.classes = cl
                self.Save(save_path)

                classes = [[] for i in range(class_number)]
                for i in data:
                    classes[self.Solve(i)].append(self.GetCloses(i))

                pyplot.figure()
                for i in classes:
                    x_ = [j[0] for j in i]
                    y_ = [j[1] for j in i]
                    pyplot.scatter(x_, y_)
                pyplot.savefig(save_path[:-4] + "png")
                pyplot.close()

    def TrainClassifierIteration(self, data, class_number, func):
        classes_1 = []
        while len(classes_1) < class_number:
            idx = random.randint(0, len(data) - 1)
            y = self.GetCloses(data[idx])
            if y not in classes_1:
                classes_1.append(y)

        b_error = 10000
        classes_2 = []
        while classes_1 != classes_2:
            classes = [[] for i in range(class_number)]
            for i in data:
                classes[self.Solve(i, classes_1)].append(self.GetCloses(i))

            error = 0
            for i in range(len(classes)):
                for j in classes[i]:
                    error += (j[0] - classes_1[i][0]) ** 2 + \
                        (j[1] - classes_1[i][1]) ** 2
            error /= len(data)

            if error < b_error:
                b_error = error

            classes_2 = [[] for i in range(len(classes_1))]

            for i in range(len(classes)):
                classes_2[i] = [sum([j[0] for j in classes[i]]) / len(classes[i]),
                                sum([j[1] for j in classes[i]]) / len(classes[i])]
            c = classes_1
            classes_1 = classes_2
            classes_2 = c
        return classes_1, b_error

    def Solve(self, x, classes_1=[]):
        if classes_1 == []:
            classes_1 = self.classes
        y = self.GetCloses(x)
        p_best = self.GetP(classes_1[0], y)
        idx_best = 0
        for j in range(len(classes_1)):
            p = self.GetP(classes_1[j], y)
            if p <= p_best:
                p_best = p
                idx_best = j
        return idx_best

    def GetP(self, x, y):
        if self.func == "sqr":
            ans = 0
            for i in range(len(x)):
                ans += (x[i] - y[i]) ** 2
            return ans
        if self.func == "8":
            ans = 0
            for i in range(len(x)):
                ans = max(ans, abs(x[i] - y[i]))
            return ans

    def GetCloses(self, x):
        dis = 5
        ans = [[0, 0]]
        for h in range(self.height):
            for w in range(self.width):
                cur_dis = Kohonin.Distance(x, self.map[h][w])
                if cur_dis == dis:
                    ans.append([w, h])
                if dis > cur_dis:
                    ans = [[w, h]]
                    dis = cur_dis
        ans = ans[random.randint(0, len(ans) - 1)]
        return ans

    def Distance(x, y):
        ans = 0
        for i in range(len(x)):
            ans += (x[i] - y[i]) ** 2
        return ans

    def GetRadius(self, t):
        return 10 * math.exp(-t * 10) * (self.width + self.height) / 2

    def Update(self, y, x, t):
        r = self.GetRadius(t)
        for h in range(self.height):
            ff = False
            for w in range(self.width):
                d = Kohonin.Distance(y, [w, h])
                if d > r:
                    if ff:
                        break
                    else:
                        continue
                ff = True
                dd = math.exp(-d) * math.exp(-t * 2)
                for i in range(self.v_size):
                    self.map[h][w][i] += dd * (x[i] - self.map[h][w][i])

    def Load(self, path):
        js = json.load(open(path))
        self.map = js["map"]
        self.classes = js["classes"]
        self.func = js["func"]

    def Save(self, path):
        json.dump({"map": self.map, "classes": self.classes,
                   "func": self.func}, open(path, "w"), indent=2)
