import json
import tqdm

data_train = []
data_test = []
count = 0

with open('data/trainset.txt') as f:
    for idx, line in enumerate(f):
        if idx % 100000 == 0:
            print('read train file line %d' % idx)
        data_train.append(json.loads(line))
        if idx == 100000:
            break

with open('data/testset.txt') as f:
    for idx, line in enumerate(f):
        data_test.append(json.loads(line))

for test in tqdm.tqdm(data_test):
    for train in data_train:
        if test == train:
            count += 1
            print("test:", test)
            print("train:", train)
print(count)