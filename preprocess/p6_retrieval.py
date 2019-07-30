import numpy as np
import json
from tqdm import tqdm
from config import METHOD, RESULT_PATH

def cal_cosine(vec1, vec2):
    return (vec1*vec2).sum()/(np.sqrt((vec1*vec1).sum())*np.sqrt((vec2*vec2).sum()))

def retrieval_with_bow():
    testset = []
    trainset = []
    cosine_maxs = []
    print("read test sentense embed...")
    with open("./data/test_bow_pro.txt", "r") as fr:
        for ids, line in enumerate(fr):
            testset.append(json.loads(line))
            if ids % 10000:
                print("read test sentense embed %d" % ids)
    print("read test sentense finish!")
    print("read train sentense embed...")
    with open("./data/train_bow_pro.txt", "r") as fr:
        for ids, line in enumerate(fr):
            trainset.append(json.loads(line))
            print("read train sentense embed %d" % ids)
    print("read train sentense finish!")
    with open(RESULT_PATH, "w") as fw:
        for test_data in tqdm(testset):
            cosines = []
            vec1 = np.array(test_data['sentense_embed'])
            for train_data in trainset:
                vec2 = np.array(train_data['sentense_embed'])
                cosine = cal_cosine(vec1, vec2)
                cosines.append(cosine)
            cosine_max = max(cosines)
            cosine_maxs.append(cosine_max)
            cosine_max_index = cosines.index(max(cosines))
            data = {'post': test_data['post'], 'response': test_data['response'],
                          'retrieval': trainset[cosine_max_index]['response'], 'cosine': cosine_max}
            fw.write(json.dumps(data) + "\n")
    print("avg cosine %f" % np.array(cosine_maxs).mean())

def retrieval_with_idf():
    testset = []
    trainset = []
    cosine_maxs = []
    print("read test sentense embed...")
    with open("./data/test_idf_pro.txt", "r") as fr:
        for ids, line in enumerate(fr):
            testset.append(json.loads(line))
            if ids % 10000:
                print("read test sentense embed %d" % ids)
    print("read test sentense finish!")
    print("read train sentense embed...")
    with open("./data/train_idf_pro.txt", "r") as fr:
        for ids, line in enumerate(fr):
            trainset.append(json.loads(line))
            print("read train sentense embed %d" % ids)
    print("read train sentense finish!")
    with open(RESULT_PATH, "w") as fw:
        for test_data in tqdm(testset):
            cosines = []
            vec1 = np.array(test_data['sentense_embed'])
            for train_data in trainset:
                vec2 = np.array(train_data['sentense_embed'])
                cosine = cal_cosine(vec1, vec2)
                cosines.append(cosine)
            cosine_max = max(cosines)
            cosine_maxs.append(cosine_max)
            cosine_max_index = cosines.index(max(cosines))
            data = {'post': test_data['post'], 'response': test_data['response'],
                          'retrieval': trainset[cosine_max_index]['response'], 'cosine': cosine_max}
            fw.write(json.dumps(data) + "\n")
    print("avg cosine %f" % np.array(cosine_maxs).mean())

def retrieval():
    if METHOD == "idf":
        retrieval_with_idf()
    else:
        retrieval_with_bow()

if __name__ == '__main__':
    retrieval()