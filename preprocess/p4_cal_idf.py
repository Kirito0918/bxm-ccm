import json
from config import VOCABULARY_SIZE
from tqdm import tqdm
import math

def load_vocabulary():
    vocab = {}
    print("load vocabulary...")
    with open("./data/vector.txt", "r") as fr:
        for ids, line in enumerate(fr):
            line = line.strip()
            word = line[: line.find(" ")]
            vector = line[line.find(" ")+1:].split()
            if len(vector) != 300:
                raise Exception("dim of vector not equal 300!")
            vector = list(map(lambda x: float(x), vector))
            vocab[word] = vector
            if ids % 10000 == 0:
                print("load vocabulary %d" % ids)
    print("load vocabulary finish!")
    if len(vocab) != VOCABULARY_SIZE:
        raise Exception("number of embed not equal VOCABULARY_SIZE!")
    return vocab

def cal_idf():
    trainset = []
    vocab = load_vocabulary()
    idfs = {}
    print("load data...")
    with open("./data/train.txt", "r") as fr:
        for ids, line in enumerate(fr):
            trainset.append(json.loads(line))
            if ids % 100000 == 0:
                print("load data %d" % ids)
    print("load data finish!")
    num_trainset = len(trainset)
    print("calculate idf...")
    for word in tqdm(vocab.keys()):
        count = 1
        for train_data in trainset:
            if word in train_data['post'] or word in train_data['response']:
                count += 1
        idf = math.log10(1.0*num_trainset/count)
        idfs[word] = idf
    print("calculate idf finish!")
    with open("./testdata/idf.txt", "w") as fw:
        fw.write(json.dumps(idfs))

if __name__ == '__main__':
    cal_idf()