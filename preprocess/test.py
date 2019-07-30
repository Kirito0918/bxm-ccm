import json
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import math

# 测试词向量
def test_vector():
    with open("../data/glove.840B.300d.txt", "r") as fr:
        for ids, line in enumerate(fr, start=1):
            line = line.split()
            word = line[0]
            vector = line[1:]
            print(word)
            print(vector)
            print(len(vector))
            if ids == 10:
                break

# 截取部分训练集
def cut_trainset():
    with open("../data/trainset.txt", "r") as fr:
        with open("./testdata/train.txt", "w") as fw:
            for ids, line in enumerate(fr, start=1):
                line = json.loads(line)
                fw.write(json.dumps(line)+"\n")
                if ids == 10000:
                    break
# 截取部分测试集
def cut_testset():
    with open("../data/testset.txt", "r") as fr:
        with open("./test.txt", "w") as fw:
            for ids, line in enumerate(fr, start=1):
                line = json.loads(line)
                fw.write(json.dumps(line)+"\n")
                if ids == 100:
                    break

# 测试测试集保存是否正确
def test_testset():
    with open("./testdata/test.txt", "r") as fr:
        for line in fr:
            line = json.loads(line)
            print(line)

# 载入预训练的词向量
def load_glove():
    embeds = {}
    with open("../data/glove.840B.300d.txt", "r", encoding="utf8") as fr:
        for index, line in enumerate(fr):
            if index % 100000 == 0:
                print("load embed %d" % index)
            line = line.strip()
            word = line[:line.find(" ")]
            embed = line[line.find(" ")+1:]
            embeds[word] = embed
    print("load embed finish!")
    return embeds

VOCABULARY_SIZE = 18000
UNK_TOKEN = "<unk>"
def get_vocabulary():
    posts = []
    responses = []
    vocabulary_dict = {}
    with open("./testdata/train.txt", "r") as fr:
        for line in fr:
            sample = json.loads(line)
            post = sample["post"]
            response = sample["response"]
            posts.append(post)
            responses.append(response)
    for post in posts:
        for word in post:
            if word in vocabulary_dict:
                vocabulary_dict[word] += 1
            else:
                vocabulary_dict[word] = 1
    for response in responses:
        for word in response:
            if word in vocabulary_dict:
                vocabulary_dict[word] += 1
            else:
                vocabulary_dict[word] = 1
    vocabulary_dict = sorted(vocabulary_dict.items(), key=lambda x: x[1], reverse=True)
    print("训练集词频：", vocabulary_dict)
    print("出现的单词个数：", len(vocabulary_dict))
    vocabulary_dict = dict([(UNK_TOKEN, 0)] + vocabulary_dict[: VOCABULARY_SIZE-1])
    print("截取词汇表大小：", len(vocabulary_dict))
    print("词汇表：", vocabulary_dict)
    return list(vocabulary_dict.keys())

def create_vocabulary():
    vocabulary = get_vocabulary()  # list
    embeds = load_glove()  # embed dict
    embed_dim_error = 0
    not_in_embed = 0
    with open("./testdata/vector.txt", "w") as fw:
        for word in vocabulary:
            if word in embeds:
                vector = embeds[word].split()
                if len(vector) != 300:
                    embed_dim_error += 1
                    not_in_embed += 1
                    vector = ['0'] * 300
            else:
                vector = ['0'] * 300
                not_in_embed += 1
            fw.write(word + " " + " ".join(vector) + "\n")
    print("create vocabulary finish!")
    print("glove dim error %d" % embed_dim_error)
    print("vocabulary not in glove %d" % not_in_embed)
    print("vocabulary not in glove rate %f" % (1.0*not_in_embed/VOCABULARY_SIZE))

def load_vocabulary():
    vocab = {}
    with open("./testdata/vector.txt", "r") as fr:
        for line in fr:
            line = line.strip()
            word = line[: line.find(" ")]
            vector = line[line.find(" ")+1:].split()
            if len(vector) != 300:
                raise Exception("dim of vector not equal 300!")
            vector = list(map(lambda x: float(x), vector))
            vocab[word] = vector
    if len(vocab) != VOCABULARY_SIZE:
        raise Exception("number of embed not equal VOCABULARY_SIZE!")
    return vocab

def process_testset():
    vocab = load_vocabulary()
    post_len = []
    not_in_vocabulary = 0
    with open("./testdata/test.txt", "r") as fr:
        with open("./testdata/test_pro.txt", "w") as fw:
            for index, line in enumerate(fr):
                line = json.loads(line)
                post = line['post']
                post_len.append(len(post))
                response = line['response']
                sentense_embed = []
                for word in post:
                    if word in vocab:
                        sentense_embed.append(vocab[word])
                    else:
                        sentense_embed.append(vocab[UNK_TOKEN])
                        not_in_vocabulary += 1
                sentense_embed = np.array(sentense_embed).mean(axis=0).tolist()
                data = {'post': post, 'response': response, 'sentense_embed': sentense_embed}
                fw.write(json.dumps(data) + "\n")
            if index % 10000 == 0:
                print("process testset %d" % index)
    print("process testset finish!")
    print("word in testset not in vocab %d" % not_in_vocabulary)
    print("word in testset not in vocab rate %f" % (1.0*not_in_vocabulary/np.array(post_len).sum()))

def process_trainset():
    vocab = load_vocabulary()
    with open("./testdata/train.txt", "r") as fr:
        with open("./testdata/train_pro.txt", "w") as fw:
            for index, line in enumerate(fr):
                line = json.loads(line)
                response = line['response']
                sentense_embed = []
                for word in response:
                    if word in vocab:
                        sentense_embed.append(vocab[word])
                    else:
                        sentense_embed.append(vocab[UNK_TOKEN])
                sentense_embed = np.array(sentense_embed).mean(axis=0).tolist()
                data = {'response': response, 'sentense_embed': sentense_embed}
                fw.write(json.dumps(data) + "\n")
            if index % 10000 == 0:
                print("process trainset %d" % index)
    print("process trainset finish!")

def cal_cosine(vec1, vec2):
    return (vec1*vec2).sum()/(np.sqrt((vec1*vec1).sum())*np.sqrt((vec2*vec2).sum()))

def retrieval_with_bow():
    testset = []
    trainset = []
    cosine_maxs = []
    with open("./testdata/test_pro.txt", "r") as fr:
        for line in fr:
            testset.append(json.loads(line))
    with open("./testdata/train_pro.txt", "r") as fr:
        for line in fr:
            trainset.append(json.loads(line))
    with open("./testdata/result_bow.txt", "w") as fw:
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

def evaluate():
    bleus, bleus1, bleus2, bleus3, bleus4 = [], [], [], [], []
    with open("./testdata/result.txt", "r") as fr:
        for line in fr:
            line = json.loads(line)
            references = [line['response']]
            hypothesis = line['retrieval']
            bleu = sentence_bleu(references, hypothesis)
            bleus.append(bleu)
            bleu1 = sentence_bleu(references, hypothesis, weights=[1, 0, 0, 0])
            bleus1.append(bleu1)
            bleu2 = sentence_bleu(references, hypothesis, weights=[0, 1, 0, 0])
            bleus2.append(bleu2)
            bleu3 = sentence_bleu(references, hypothesis, weights=[0, 0, 1, 0])
            bleus3.append(bleu3)
            bleu4 = sentence_bleu(references, hypothesis, weights=[0, 0, 0, 1])
            bleus4.append(bleu4)
    print("avg bleu %f" % np.array(bleus).mean())
    print("avg bleu-1 %f" % np.array(bleus1).mean())
    print("avg bleu-2 %f" % np.array(bleus2).mean())
    print("avg bleu-3 %f" % np.array(bleus3).mean())
    print("avg bleu-4 %f" % np.array(bleus4).mean())

def cal_idf():
    trainset = []
    vocab = load_vocabulary()
    idfs = {}
    with open("./testdata/train.txt", "r") as fr:
        for line in fr:
            trainset.append(json.loads(line))
    num_trainset = len(trainset)
    for word in tqdm(vocab.keys()):
        count = 1
        for train_data in trainset:
            if word in train_data['post'] or word in train_data['response']:
                count += 1
        idf = math.log10(1.0*num_trainset/count)
        idfs[word] = idf
    with open("./testdata/idf.txt", "w") as fw:
        fw.write(json.dumps(idfs))



if __name__ == '__main__':
    # cut_trainset()
    # cut_testset()
    # test_testset()
    # get_vocabulary()
    # create_vocabulary()
    # load_vocabulary()
    # process_testset()
    # process_trainset()
    # retrieval_with_bow()
    # evaluate()
    cal_idf()
    pass
