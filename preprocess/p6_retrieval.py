import numpy as np
import json
from tqdm import tqdm
from config import METHOD, RESULT_PATH, RETRIEVAL_SCOPE

# def cal_cosine(vec1, vec2):
#     return (vec1*vec2).sum()/(np.sqrt((vec1*vec1).sum())*np.sqrt((vec2*vec2).sum()))
#
# def retrieval_with_bow():
#     testset = []
#     trainset = []
#     cosine_maxs = []
#     print("read test sentence embed...")
#     with open("./data/test_bow_pro.txt", "r") as fr:
#         for ids, line in enumerate(fr):
#             testset.append(json.loads(line))
#             if ids % 10000 == 0:
#                 print("read test sentence embed %d" % ids)
#     print("read test sentence finish!")
#     print("read train sentence embed...")
#     with open("./data/train_bow_pro.txt", "r") as fr:
#         for ids, line in enumerate(fr):
#             trainset.append(json.loads(line))
#             if ids % 100000 == 0:
#                 print("read train sentence embed %d" % ids)
#     print("read train sentence finish!")
#     with open(RESULT_PATH, "w") as fw:
#         for test_data in tqdm(testset):
#             cosines = []
#             vec1 = np.array(test_data['sentence_embed'])
#             for train_data in trainset:
#                 vec2 = np.array(train_data['sentence_embed'])
#                 cosine = cal_cosine(vec1, vec2)
#                 cosines.append(cosine)
#             cosine_max = max(cosines)
#             cosine_maxs.append(cosine_max)
#             cosine_max_index = cosines.index(max(cosines))
#             data = {'post': test_data['post'], 'response': test_data['response'],
#                           'retrieval': trainset[cosine_max_index]['response'], 'cosine': cosine_max}
#             fw.write(json.dumps(data) + "\n")
#     print("avg cosine %f" % np.array(cosine_maxs).mean())
#
# def retrieval_with_idf():
#     testset = []
#     trainset = []
#     cosine_maxs = []
#     print("read test sentence embed...")
#     with open("./data/test_idf_pro.txt", "r") as fr:
#         for ids, line in enumerate(fr):
#             testset.append(json.loads(line))
#             if ids % 10000 == 0:
#                 print("read test sentence embed %d" % ids)
#     print("read test sentence finish!")
#     print("read train sentence embed...")
#     with open("./data/train_idf_pro.txt", "r") as fr:
#         for ids, line in enumerate(fr):
#             trainset.append(json.loads(line))
#             if ids % 100000 == 0:
#                 print("read train sentence embed %d" % ids)
#     print("read train sentence finish!")
#     with open(RESULT_PATH, "w") as fw:
#         for test_data in tqdm(testset):
#             cosines = []
#             vec1 = np.array(test_data['sentence_embed'])
#             for train_data in trainset:
#                 vec2 = np.array(train_data['sentence_embed'])
#                 cosine = cal_cosine(vec1, vec2)
#                 cosines.append(cosine)
#             cosine_max = max(cosines)
#             cosine_maxs.append(cosine_max)
#             cosine_max_index = cosines.index(max(cosines))
#             data = {'post': test_data['post'], 'response': test_data['response'],
#                     'retrieval': trainset[cosine_max_index]['response'], 'cosine': cosine_max}
#             fw.write(json.dumps(data) + "\n")
#     print("avg cosine %f" % np.array(cosine_maxs).mean())

def cal_cosine(vec1, vec2):
    return ((vec1*vec2).sum(axis=1)) / (np.sqrt((vec1*vec1).sum(axis=1)) * np.sqrt((vec2*vec2).sum(axis=1)))

def retrieval_with_bow():
    testset = []
    trainset = []
    cosine_maxs = []
    train_sentence_embed = []
    print("read test sentence embed...")
    with open("./data/test_bow_pro.txt", "r") as fr:
        for ids, line in enumerate(fr):
            testset.append(json.loads(line))
            if ids % 10000 == 0:
                print("read test sentence embed %d" % ids)
    print("read test sentence finish!")
    print("read train sentence embed...")
    with open("./data/train_bow_pro.txt", "r") as fr:
        for ids, line in enumerate(fr, start=1):
            line = json.loads(line)
            trainset.append(line)
            train_sentence_embed.append(line['sentence_embed'])
            if ids % 100000 == 0:
                print("read train sentence embed %d" % ids)
            if ids == RETRIEVAL_SCOPE:
                break
    print("read train sentence finish!")
    train_sentence_embed = np.array(train_sentence_embed)
    len_trainset = len(trainset)
    with open(RESULT_PATH, "w") as fw:
        for test_data in tqdm(testset):
            test_sentence_embed = np.array([test_data['sentence_embed']*len_trainset]).reshape((len_trainset, 300))
            cosines = cal_cosine(train_sentence_embed, test_sentence_embed).tolist()
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
    train_sentence_embed = []
    print("read test sentence embed...")
    with open("./data/test_idf_pro.txt", "r") as fr:
        for ids, line in enumerate(fr):
            testset.append(json.loads(line))
            if ids % 10000 == 0:
                print("read test sentence embed %d" % ids)
    print("read test sentence finish!")
    print("read train sentence embed...")
    with open("./data/train_idf_pro.txt", "r") as fr:
        for ids, line in enumerate(fr, start=1):
            line = json.loads(line)
            trainset.append(line)
            train_sentence_embed.append(line['sentence_embed'])
            if ids % 100000 == 0:
                print("read train sentence embed %d" % ids)
            if ids == RETRIEVAL_SCOPE:
                break
    print("read train sentence finish!")
    train_sentence_embed = np.array(train_sentence_embed)
    len_trainset = len(trainset)
    with open(RESULT_PATH, "w") as fw:
        for test_data in tqdm(testset):
            test_sentence_embed = np.array([test_data['sentence_embed'] * len_trainset]).reshape((len_trainset, 300))
            cosines = cal_cosine(train_sentence_embed, test_sentence_embed).tolist()
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