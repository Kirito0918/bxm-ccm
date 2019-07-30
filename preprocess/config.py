import sys

NUM_TRAIN = 20000  # sys.maxsize  # 截取训练集长度
NUM_TEST = 1000  # sys.maxsize  # 截取测试集长度
VOCABULARY_SIZE = 50000  # 选取词汇表大小
UNK_TOKEN = "<unk>"  # 词汇表外单词
METHOD = "bow"  # 句子向量计算方式
RESULT_PATH = "./data/result_bow.txt"