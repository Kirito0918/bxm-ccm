TESTSET_PATH = '../data/testset.txt'
RETRIEVAL_RESULT = './data/result/result_bow.txt'
RESULT_PATH = './data/result/test_set.txt'

import json
from tqdm import tqdm

ft = open(TESTSET_PATH, 'r', encoding='utf8')
fr = open(RETRIEVAL_RESULT, 'r', encoding='utf8')
fw = open(RESULT_PATH, 'w', encoding='utf8')

head = fr.readline()

for _ in tqdm(range(20000)):
    try:
        tline = ft.readline()
        rline = fr.readline()
        testset = json.loads(tline)
        retrieval = json.loads(rline)
        testset['retrieval'] = retrieval['retrieval']
        fw.write(json.dumps(testset) + '\n')
    except Exception as e:
        print(tline)
        print(rline)
        print(_)
        break

ft.close()
fr.close()
fw.close()

