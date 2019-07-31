import sys

NUM_TRAIN = sys.maxsize  # sys.maxsize  # 截取训练集长度
NUM_TEST = sys.maxsize  # sys.maxsize  # 截取测试集长度
RETRIEVAL_SCOPE = 200000  # 从多少训练集中去检索
VOCABULARY_SIZE = 50000  # 选取词汇表大小
UNK_TOKEN = "<unk>"  # 词汇表外单词
METHOD = "bow"  # 句子向量计算方式
RESULT_PATH = "./data/result/result_bow"