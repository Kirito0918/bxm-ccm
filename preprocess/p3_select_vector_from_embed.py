import json
from config import UNK_TOKEN
from config import VOCABULARY_SIZE
from tqdm import tqdm

# 选取词汇表
def get_vocabulary():
    posts = []
    responses = []
    vocabulary_dict = {}
    print("load data...")
    with open("./data/train.txt", "r") as fr:
        for ids, line in enumerate(fr):
            sample = json.loads(line)
            post = sample["post"]
            response = sample["response"]
            posts.append(post)
            responses.append(response)
            if ids % 100000 == 0:
                print("load data %d" % ids)
    print("load data finish!")
    print("calculate the frequency of word from post")
    for post in tqdm(posts):
        for word in post:
            if word in vocabulary_dict:
                vocabulary_dict[word] += 1
            else:
                vocabulary_dict[word] = 1
    print("calculate the frequency of word from response")
    for response in responses:
        for word in response:
            if word in vocabulary_dict:
                vocabulary_dict[word] += 1
            else:
                vocabulary_dict[word] = 1
    print("calculate the frequency of word finish!")
    vocabulary_dict = sorted(vocabulary_dict.items(), key=lambda x: x[1], reverse=True)
    print("训练集词频：", vocabulary_dict)
    print("出现的单词个数：", len(vocabulary_dict))
    vocabulary_dict = dict([(UNK_TOKEN, 0)] + vocabulary_dict[: VOCABULARY_SIZE-1])
    print("截取词汇表大小：", len(vocabulary_dict))
    print("词汇表：", vocabulary_dict)
    return list(vocabulary_dict.keys())

# 载入预训练的词向量
def load_glove():
    embeds = {}
    print("load embed...")
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

# 从预训练词向量中选取词汇表中所需要的
def create_vocabulary():
    vocabulary = get_vocabulary()  # list
    embeds = load_glove()  # embed dict
    embed_dim_error = 0  # 词向量中是否有维度错误的
    not_in_embed = 0  # 词汇表中有多少没有预训练的词向量
    print("create vocabulary...")
    with open("./data/vector.txt", "w") as fw:
        for word in tqdm(vocabulary):
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

if __name__ == '__main__':
    create_vocabulary()