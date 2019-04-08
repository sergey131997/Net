import json
import random
import sys

if __name__ == "__main__":
    path = sys.argv[1]
    js = json.load(open(path))
    js_train = [[] for i in range(len(js))]
    js_test = [[] for i in range(len(js))]
    for i in range(len(js)):
        for j in range(len(js[i])):
            if random.random() < 0.3:
                js_test[i].append(js[i][j])
            else:
                js_train[i].append(js[i][j])
    json.dump(js_test, open(path[:-4] + "_test.json", "w"), indent=2)
    json.dump(js_train, open(path[:-4] + "_train.json", "w"), indent=2)
