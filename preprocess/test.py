import json
import numpy as np

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
        with open("./train.txt", "w") as fw:
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
    with open("./test.txt", "r") as fr:
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
    with open("./train.txt", "r") as fr:
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
    with open("vector", "w") as fw:
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
    vocabulary = {}
    with open("./vector", "r") as fr:
        for line in fr:
            line = line.strip()
            word = line[: line.find(" ")]
            vector = line[line.find(" ")+1:].split()
            if len(vector) != 300:
                raise Exception("dim of vector not equal 300!")
            vector = np.array(map(lambda x: float(x), vector))
            vocabulary[word] = vector
    if len(vocabulary) != VOCABULARY_SIZE:
        raise Exception("number of embed not equal VOCABULARY_SIZE!")
    print(vocabulary)




if __name__ == '__main__':
    # cut_trainset()
    # cut_testset()
    # test_testset()
    # get_vocabulary()
    # create_vocabulary()
    load_vocabulary()
    pass
