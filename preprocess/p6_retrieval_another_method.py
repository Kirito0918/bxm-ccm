import numpy as np
import json
from tqdm import tqdm
from config import METHOD, RESULT_PATH, RETRIEVAL_SCOPE
import sys

def cal_cosine(vec1, vec2):
    return (vec1*vec2).sum()/(np.sqrt((vec1*vec1).sum())*np.sqrt((vec2*vec2).sum()))

def retrieval_with_bow():
    testset = []
    cosine_maxs = []
    print("read test sentence embed...")
    with open("./data/test_bow_pro.txt", "r") as fr:
        for ids, line in enumerate(fr):
            testset.append(json.loads(line))
            if ids % 10000 == 0:
                print("read test sentence embed %d" % ids)
    print("read test sentence finish!")
    with open(RESULT_PATH, "w") as fw:
        for test_data in tqdm(testset):
            cosine_max = -sys.maxsize
            response_max = []
            vec1 = np.array(test_data['sentence_embed'])
            with open("./data/train_bow_pro.txt", "r") as fr:
                for line in fr:
                    line = json.loads(line)
                    vec2 = line['sentence_embed']
                    cosine = cal_cosine(vec1, vec2)
                    if cosine > cosine_max:
                        cosine_max = cosine
                        response_max = line['response']
            cosine_maxs.append(cosine_max)
            data = {'post': test_data['post'], 'response': test_data['response'],
                          'retrieval': response_max, 'cosine': cosine_max}
            fw.write(json.dumps(data) + "\n")
    print("avg cosine %f" % np.array(cosine_maxs).mean())

def retrieval_with_idf():
    testset = []
    cosine_maxs = []
    print("read test sentence embed...")
    with open("./data/test_idf_pro.txt", "r") as fr:
        for ids, line in enumerate(fr):
            testset.append(json.loads(line))
            if ids % 10000 == 0:
                print("read test sentence embed %d" % ids)
    print("read test sentence finish!")
    with open(RESULT_PATH, "w") as fw:
        for test_data in tqdm(testset):
            cosine_max = -sys.maxsize
            response_max = []
            vec1 = np.array(test_data['sentence_embed'])
            with open("./data/train_idf_pro.txt", "r") as fr:
                for line in fr:
                    line = json.loads(line)
                    vec2 = line['sentence_embed']
                    cosine = cal_cosine(vec1, vec2)
                    if cosine > cosine_max:
                        cosine_max = cosine
                        response_max = line['response']
            cosine_maxs.append(cosine_max)
            data = {'post': test_data['post'], 'response': test_data['response'],
                          'retrieval': response_max, 'cosine': cosine_max}
            fw.write(json.dumps(data) + "\n")
    print("avg cosine %f" % np.array(cosine_maxs).mean())

def retrieval():
    if METHOD == "idf":
        retrieval_with_idf()
    else:
        retrieval_with_bow()

if __name__ == '__main__':
    retrieval()