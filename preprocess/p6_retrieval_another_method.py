import numpy as np
import json
from tqdm import tqdm
from config import METHOD, RESULT_PATH, RETRIEVAL_SCOPE
import sys

def get_num_epoch():
    lines = []
    with open("./data/train.txt", "r") as fr:
        for line in fr:
            lines.append(1)
    num_train = len(lines)
    num_epoch = (num_train // RETRIEVAL_SCOPE)
    if num_train % (num_epoch * RETRIEVAL_SCOPE) != 0:
        num_epoch += 1
    print("训练集长度%d，以%d个样本为一轮，共需要%d轮" % (num_train, RETRIEVAL_SCOPE, num_epoch))
    return num_epoch

# def cal_cosine(vec1, vec2):
#     return ((vec1*vec2).sum(axis=1)) / (np.sqrt((vec1*vec1).sum(axis=1)) * np.sqrt((vec2*vec2).sum(axis=1)))

def cal_cosine(vec1, vec2):
    return (vec1*vec2).sum()/(np.sqrt((vec1*vec1).sum())*np.sqrt((vec2*vec2).sum()))

def retrieval_with_bow():
    num_epoch = get_num_epoch()  # 要检索的轮数
    testset = []  # 测试集
    print("read test sentence embed...")
    with open("./data/test_bow_pro.txt", "r") as fr:
        for ids, line in enumerate(fr):
            testset.append(json.loads(line))
            if ids % 10000 == 0:
                print("read test sentence embed %d" % ids)
    print("read test sentence finish!")
    num_test = len(testset)
    cosine_maxs = [-sys.maxsize] * num_test  # 最大的cosine相似度
    response_maxs = [[""]] * num_test  # cosine相似度最大的回复

    for epoch in range(num_epoch):
        print("start epoch %d retrieval" % epoch)
        start = epoch*RETRIEVAL_SCOPE
        end = (epoch+1)*RETRIEVAL_SCOPE
        trainset_embed = []
        trainset_response = []
        with open("./data/train_bow_pro.txt", "r") as fr:
            for ids, line in enumerate(fr):
                if ids >= start and ids < end:
                    line = json.loads(line)
                    trainset_embed.append(line['sentence_embed'])
                    trainset_response.append(line['response'])
                if ids >= end:
                    break
        num_trainset = len(trainset_response)
        trainset_embed = np.array(trainset_embed)
        print("read train sentence finish! read %d lines!" % num_trainset)

        for ids in tqdm(range(num_test)):
            cosines = []
            vec1 = np.array(testset[ids]['sentence_embed'])
            for idx in range(num_trainset):
                vec2 = trainset_embed[idx]
                cosine = cal_cosine(vec1, vec2)
                cosines.append(cosine)
            cosine_max = max(cosines)
            cosine_max_index = cosines.index(cosine_max)
            if cosine_max > cosine_maxs[ids]:
                cosine_maxs[ids] = cosine_max
                response_maxs[ids] = trainset_response[cosine_max_index]
        print("avg cosine %f" % np.array(cosine_maxs).mean())

        # for ids in tqdm(range(num_test)):
        #     testset_embed = np.array([testset[ids]['sentence_embed']*num_trainset]).reshape((num_trainset, 300))
        #     cosines = cal_cosine(trainset_embed, testset_embed).tolist()
        #     cosine_max = max(cosines)
        #     cosine_max_index = cosines.index(cosine_max)
        #     if cosine_max > cosine_maxs[ids]:
        #         cosine_maxs[ids] = cosine_max
        #         response_maxs[ids] = trainset_response[cosine_max_index]
        # print("avg cosine %f" % np.array(cosine_maxs).mean())

        with open(RESULT_PATH + "_epoch_" + str(epoch) + ".txt", "w") as fw:
            fw.write("epoch %d, avg cosine %f\n" % (epoch, np.array(cosine_maxs).mean()))
            for ids, test_data in enumerate(testset):
                data = {'post': test_data['post'], 'response': test_data['response'],
                        'retrieval': response_maxs[ids], 'cosine': cosine_maxs[ids]}
                fw.write(json.dumps(data) + "\n")

        print("finish epoch %d retrieval!" % epoch)
    print("avg cosine %f" % np.array(cosine_maxs).mean())

def retrieval_with_idf():
    num_epoch = get_num_epoch()  # 要检索的轮数
    testset = []  # 测试集
    print("read test sentence embed...")
    with open("./data/test_idf_pro.txt", "r") as fr:
        for ids, line in enumerate(fr):
            testset.append(json.loads(line))
            if ids % 10000 == 0:
                print("read test sentence embed %d" % ids)
    print("read test sentence finish!")
    num_test = len(testset)
    cosine_maxs = [-sys.maxsize] * num_test  # 最大的cosine相似度
    response_maxs = [[""]] * num_test  # cosine相似度最大的回复

    for epoch in range(num_epoch):
        print("start epoch %d retrieval" % epoch)
        start = epoch*RETRIEVAL_SCOPE
        end = (epoch+1)*RETRIEVAL_SCOPE
        trainset_embed = []
        trainset_response = []
        with open("./data/train_idf_pro.txt", "r") as fr:
            for ids, line in enumerate(fr):
                if ids >= start and ids < end:
                    line = json.loads(line)
                    trainset_embed.append(line['sentence_embed'])
                    trainset_response.append(line['response'])
                if ids >= end:
                    break
        num_trainset = len(trainset_response)
        trainset_embed = np.array(trainset_embed)
        print("read train sentence finish! read %d lines!" % num_trainset)

        for ids in tqdm(range(num_test)):
            cosines = []
            vec1 = np.array(testset[ids]['sentence_embed'])
            for idx in range(num_trainset):
                vec2 = trainset_embed[idx]
                cosine = cal_cosine(vec1, vec2)
                cosines.append(cosine)
            cosine_max = max(cosines)
            cosine_max_index = cosines.index(cosine_max)
            if cosine_max > cosine_maxs[ids]:
                cosine_maxs[ids] = cosine_max
                response_maxs[ids] = trainset_response[cosine_max_index]
        print("avg cosine %f" % np.array(cosine_maxs).mean())

        # for ids in tqdm(range(num_test)):
        #     testset_embed = np.array([testset[ids]['sentence_embed']*num_trainset]).reshape((num_trainset, 300))
        #     cosines = cal_cosine(trainset_embed, testset_embed).tolist()
        #     cosine_max = max(cosines)
        #     cosine_max_index = cosines.index(cosine_max)
        #     if cosine_max > cosine_maxs[ids]:
        #         cosine_maxs[ids] = cosine_max
        #         response_maxs[ids] = trainset_response[cosine_max_index]
        # print("avg cosine %f" % np.array(cosine_maxs).mean())

        with open(RESULT_PATH + "_epoch_" + str(epoch) + ".txt", "w") as fw:
            fw.write("epoch %d, avg cosine %f\n" % (epoch, np.array(cosine_maxs).mean()))
            for ids, test_data in enumerate(testset):
                data = {'post': test_data['post'], 'response': test_data['response'],
                        'retrieval': response_maxs[ids], 'cosine': cosine_maxs[ids]}
                fw.write(json.dumps(data) + "\n")

        print("finish epoch %d retrieval!" % epoch)
    print("avg cosine %f" % np.array(cosine_maxs).mean())

def retrieval():
    if METHOD == "idf":
        retrieval_with_idf()
    else:
        retrieval_with_bow()

if __name__ == '__main__':
    retrieval()