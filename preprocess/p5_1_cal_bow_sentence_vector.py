from config import VOCABULARY_SIZE, UNK_TOKEN
import json
import numpy as np

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
    return vocab

def process_trainset():
    vocab = load_vocabulary()
    print("process trainset...")
    with open("./data/train.txt", "r") as fr:
        with open("./data/train_bow_pro.txt", "w") as fw:
            for ids, line in enumerate(fr):
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
            if ids % 100000 == 0:
                print("process trainset %d" % ids)
    print("process trainset finish!")

def process_testset():
    vocab = load_vocabulary()
    post_len = []
    not_in_vocabulary = 0  # 测试集post中单词不存在于词汇表的数量
    print("process testset...")
    with open("./data/test.txt", "r") as fr:
        with open("./data/test_bow_pro.txt", "w") as fw:
            for ids, line in enumerate(fr):
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
            if ids % 10000 == 0:
                print("process testset %d" % ids)
    print("process testset finish!")
    print("word in testset not in vocab %d" % not_in_vocabulary)
    print("word in testset not in vocab rate %f" % (1.0*not_in_vocabulary/np.array(post_len).sum()))

def cal_bow_embed():
    process_trainset()
    process_testset()

if __name__ == '__main__':
    cal_bow_embed()