import os
import json

data_train = []
data_test = []

flag = False
with open('data/trainset.txt') as f:
    for idx, line in enumerate(f):
        if idx % 100000 == 0:
            print('read train file line %d' % idx)
        data_train.append(json.loads(line))
        if idx == 300000:  # 用来删减数据集
            break

with open('data/testset.txt') as f:
    for idx, line in enumerate(f):
        data_test.append(json.loads(line))

for test in data_test:
    for train in data_train:
        if test == train:
            print("test:", test)
            print("train:", train)