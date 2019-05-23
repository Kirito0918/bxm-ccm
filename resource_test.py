import json

with open('./data/resource.txt') as f:
    d = json.loads(f.readline())

csk_triples = d['csk_triples']  # 三元组列表 ["实体,关系,实体",...]
csk_entities = d['csk_entities']  # 实体列表 ["实体",...]
raw_vocab = d['vocab_dict']  # 词汇表，是一个字典
kb_dict = d['dict_csk']  # 知识图字典 {"实体"：["实体,关系,实体" ,...}

data_train = []
with open('./data/trainset.txt') as f:
    for idx, line in enumerate(f):
        if idx % 1000 == 0:
            print('read train file line %d' % idx)
        data_train.append(json.loads(line))
        if idx == 1000:  # 用来删减数据集
            break

# print([[csk_triples[x].split(', ') for x in triple] for triple in data_train[0]['all_triples']])

with open('./data/mytest.txt') as f:
    pass



