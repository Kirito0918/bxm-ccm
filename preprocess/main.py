from config import METHOD
from p1_processs_data import cut_trainset, cut_testset
from p2_create_vocabulary import get_vocabulary
from p3_select_vector_from_embed import create_vocabulary
from p4_cal_idf import cal_idf
from p5_1_cal_bow_sentense_vector import cal_bow_embed
from p5_2_cal_idf_sentense_vector import cal_idf_embed
from p6_retrieval import retrieval_with_bow, retrieval_with_idf
from p7_evaluate import evaluate

def main():
    print("####################开始p1的处理####################")
    print("开始截取训练集")
    cut_trainset()
    print("截取训练集完成！")
    print("开始截取测试集")
    cut_testset()
    print("截取测试集完成！")
    print("####################完成p1的处理####################")
    print("####################开始p2的处理####################")
    print("开始统计词频")
    get_vocabulary()
    print("统计词频完成，根据实际情况修改词汇表大小后重新运行！")
    print("####################完成p2的处理####################")
    print("####################开始p3的处理####################")
    print("开始从预训练的词嵌入中选择词汇表需要的嵌入")
    create_vocabulary()
    print("词嵌入的选择完成！")
    print("####################完成p3的处理####################")
    print("####################开始p4的处理####################")
    if METHOD == "idf":
        print("开始计算idf")
        cal_idf()
        print("计算idf完成！")
        print("####################完成p4的处理####################")
    else:
        print("bow方式不需要这一步处理")
        print("####################跳过p4的处理####################")
    print("####################开始p5的处理####################")
    print("开始计算句子向量")
    if METHOD == "idf":
        cal_idf_embed()
    else:
        cal_bow_embed()
    print("计算句子向量完成！")
    print("####################完成p5的处理####################")
    print("####################开始p6的处理####################")
    print("开始检索句子")
    if METHOD == "idf":
        retrieval_with_idf()
    else:
        retrieval_with_bow()
    print("检索句子向量完成！")
    print("####################完成p6的处理####################")
    print("####################开始p7的处理####################")
    print("开始评估结果")
    evaluate()
    print("评估完成！")
    print("####################完成p7的处理####################")

if __name__ == '__main__':
    main()