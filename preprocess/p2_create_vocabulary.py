import json
from config import UNK_TOKEN
from config import VOCABULARY_SIZE
from tqdm import tqdm

# 统计词频，从而来帮助调整词汇表大小，如果不想改词汇表大小可以跳过步骤
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
    for post in tqdm(posts):
        for word in post:
            if word in vocabulary_dict:
                vocabulary_dict[word] += 1
            else:
                vocabulary_dict[word] = 1
    for response in tqdm(responses):
        for word in response:
            if word in vocabulary_dict:
                vocabulary_dict[word] += 1
            else:
                vocabulary_dict[word] = 1
    print("calculate the frequency of word finish!")
    vocabulary_dict = sorted(vocabulary_dict.items(), key=lambda x: x[1], reverse=True)
    # print("训练集词频：", vocabulary_dict)
    print("出现的单词个数：", len(vocabulary_dict))
    vocabulary_dict = dict([(UNK_TOKEN, 0)] + vocabulary_dict[: VOCABULARY_SIZE-1])
    print("截取词汇表大小：", len(vocabulary_dict))
    # print("词汇表：", vocabulary_dict)
    return list(vocabulary_dict.keys())

if __name__ == '__main__':
    get_vocabulary()