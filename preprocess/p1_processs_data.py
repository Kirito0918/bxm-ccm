import json
from config import NUM_TRAIN
from config import NUM_TEST

# 截取数据集
def cut_trainset():
    print("process trainset...")
    with open("../data/trainset.txt", "r") as fr:
        with open("./data/train.txt", "w") as fw:
            for ids, line in enumerate(fr, start=1):
                line = json.loads(line)
                data = {'post': line['post'], 'response': line['response']}
                fw.write(json.dumps(data)+"\n")
                if ids % 100000 == 0:
                    print("process %d" % ids)
                if ids == NUM_TRAIN:
                    break
    print("process trainset finish!")

def cut_testset():
    print("process testset...")
    with open("../data/testset.txt", "r") as fr:
        with open("./data/test.txt", "w") as fw:
            for ids, line in enumerate(fr, start=1):
                line = json.loads(line)
                data = {'post': line['post'], 'response': line['response']}
                fw.write(json.dumps(data)+"\n")
                if ids % 10000 == 0:
                    print("process %d" % ids)
                if ids == NUM_TEST:
                    break
    print("process testset finish!")

if __name__ == '__main__':
    cut_trainset()
    cut_testset()