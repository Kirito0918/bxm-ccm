import json
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from config import RESULT_PATH

def evaluate():
    bleus, bleus1, bleus2, bleus3, bleus4 = [], [], [], [], []
    with open(RESULT_PATH + '.txt', "r") as fr:
        for ids, line in enumerate(fr):
            if ids == 0:
                continue
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
            if ids % 10000 == 0:
                print("evaluate %d" % ids)
    print("avg bleu %f" % np.array(bleus).mean())
    print("avg bleu-1 %f" % np.array(bleus1).mean())
    print("avg bleu-2 %f" % np.array(bleus2).mean())
    print("avg bleu-3 %f" % np.array(bleus3).mean())
    print("avg bleu-4 %f" % np.array(bleus4).mean())

if __name__ == '__main__':
    evaluate()