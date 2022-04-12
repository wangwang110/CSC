#
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import re


def sent_mertic(all_srcs, all_pres, all_trgs):
    """
    通过模型输出文本和标准文本计算指标
    :param all_pres:
    :param all_trgs:
    :return:
    """

    sen_acc = 0
    setsum = 0
    sen_mod = 0
    sen_mod_acc = 0
    sen_tar_mod = 0
    d_sen_acc = 0
    d_sen_mod = 0
    d_sen_mod_acc = 0
    d_sen_tar_mod = 0

    for s, p, t in zip(all_srcs, all_pres, all_trgs):
        if len(s) != len(p) or len(t) != len(p):
            print(s)
            print(p)
            print(t)
            print("\n\n")

    input = [list(line) for line in all_srcs]
    out = [list(line) for line in all_pres]
    output = [list(line) for line in all_trgs]

    mod_sen = [1 if out[i] != input[i] else 0 for i in range(len(out))]  # 修改过的句子
    acc_sen = [1 if out[i] == output[i] else 0 for i in range(len(out))]  # 输出正确的句子，不一定做了修改
    tar_sen = [1 if output[i] != input[i] else 0 for i in range(len(out))]  # 实际有错的句子

    sen_mod += sum(mod_sen)
    sen_mod_acc += sum(np.multiply(np.array(mod_sen), np.array(acc_sen)))  # 修改过并且正确的句子
    sen_tar_mod += sum(tar_sen)
    sen_acc += sum(acc_sen)  # 用于算准确度
    setsum += len(output)

    prob_ = [[0 if input[i][j] == out[i][j] else 1 for j in range(len(input[i]))] for i in range(len(input))]
    label = [[0 if input[i][j] == output[i][j] else 1 for j in range(len(input[i]))] for i in range(len(input))]

    d_acc_sen = [1 if prob_[i] == label[i] else 0 for i in range(len(prob_))]  # 句子是否检测正确
    d_mod_sen = [0 if sum(prob_[i]) == 0 else 1 for i in range(len(prob_))]  # 句子是否做了修改
    d_tar_sen = [0 if sum(label[i]) == 0 else 1 for i in range(len(label))]  # 句子是否有误
    d_sen_mod += sum(d_mod_sen)
    d_sen_mod_acc += sum(np.multiply(np.array(d_mod_sen), np.array(d_acc_sen)))  # 句子有错误，并且检测对了
    d_sen_tar_mod += sum(d_tar_sen)
    d_sen_acc += sum(d_acc_sen)

    print(d_sen_mod)
    print(d_sen_tar_mod)

    print(sen_mod)
    print(sen_tar_mod)

    d_precision = d_sen_mod_acc / d_sen_mod
    d_recall = d_sen_mod_acc / d_sen_tar_mod
    d_F1 = 2 * d_precision * d_recall / (d_precision + d_recall)

    c_precision = sen_mod_acc / sen_mod
    c_recall = sen_mod_acc / sen_tar_mod
    c_F1 = 2 * c_precision * c_recall / (c_precision + c_recall)
    print(
        "detection sentence accuracy:{0},p:{1},r:{2},F1:{3}".format(round(d_sen_acc / setsum, 3), round(d_precision, 3),
                                                                    round(d_recall, 3), round(d_F1, 3)))
    print("correction sentence accuracy:{0},p:{1},recall:{2},F1:{3}".format(round(sen_acc / setsum, 3),
                                                                            round(sen_mod_acc / sen_mod, 3),
                                                                            round(sen_mod_acc / sen_tar_mod, 3),
                                                                            round(c_F1, 3)))
    print("sentence target modify:{0},sentence sum:{1},sentence modified accurate:{2}".format(sen_tar_mod, setsum,
                                                                                              sen_mod_acc))
    # accuracy, precision, recall, F1
    return sen_acc / setsum, sen_mod_acc / sen_mod, sen_mod_acc / sen_tar_mod, c_F1


def token_mertic(all_srcs, all_pres, all_trgs):
    """
        通过模型输出文本和标准文本计算指标
        :param all_pres:
        :param all_trgs:
        :return:
    """

    sen_acc = 0
    setsum = 0
    sen_mod = 0
    sen_mod_acc = 0
    sen_tar_mod = 0
    d_sen_acc = 0
    d_sen_mod = 0
    d_sen_mod_acc = 0
    d_sen_tar_mod = 0

    input = [list(line) for line in all_srcs]
    out = [list(line) for line in all_pres]
    output = [list(line) for line in all_trgs]

    mod_sen = [[0 if input[i][j] == out[i][j] else 1 for j in range(len(input[i]))] for i in range(len(input))]
    # 模型修改过的token
    acc_sen = [[1 if out[i][j] == output[i][j] else 0 for j in range(len(input[i]))] for i in range(len(input))]
    # 模型输出对了的token (不一定做了修改)
    tar_sen = [[0 if input[i][j] == output[i][j] else 1 for j in range(len(input[i]))] for i in range(len(input))]
    # 实际要修改的token

    sen_mod += sum([sum(mod_sen[i]) for i in range(len(input))])  # 修改过的token

    sen_mod_acc += sum(
        [sum(np.multiply(np.array(mod_sen[i]), np.array(acc_sen[i]))) for i in range(len(input))])  # 修改过并且正确的token

    sen_tar_mod += sum([sum(tar_sen[i]) for i in range(len(input))])  # 实际有错的token
    sen_acc += sum([sum(acc_sen[i]) for i in range(len(input))])  # 用于算准确度,token二分类检测是否有错
    setsum += sum([len(input[i]) for i in range(len(input))])

    # 实际错误的token
    # 检测对了的 （检测到错误的token，里面实际错误的token）
    # 检测到错误的token

    # prob_ = [[0 if input[i][j] == out[i][j] else 1 for j in range(len(input[i]))] for i in range(len(input))]
    # label = [[0 if input[i][j] == output[i][j] else 1 for j in range(len(input[i]))] for i in range(len(input))]

    d_acc_sen = [[1 if mod_sen[i][j] == tar_sen[i][j] else 0 for j in range(len(input[i]))] for i in
                 range(len(input))]  # token是否检测正确(二分类)
    d_mod_sen = mod_sen

    d_tar_sen = tar_sen

    d_sen_mod += sum([sum(d_mod_sen[i]) for i in range(len(input))])  # 检测到错误的token
    d_sen_mod_acc += sum(
        [sum(np.multiply(np.array(d_mod_sen[i]), np.array(d_acc_sen[i]))) for i in range(len(input))])
    # 检测对了的 （检测到错误的token，里面实际错误的token）
    d_sen_tar_mod += sum([sum(d_tar_sen[i]) for i in range(len(input))])
    # 实际错误的token
    d_sen_acc += sum([sum(d_acc_sen[i]) for i in range(len(input))])

    print(d_sen_mod)
    print(d_sen_tar_mod)

    print(sen_mod)
    print(sen_tar_mod)

    d_precision = d_sen_mod_acc / d_sen_mod
    d_recall = d_sen_mod_acc / d_sen_tar_mod
    d_F1 = 2 * d_precision * d_recall / (d_precision + d_recall)

    c_precision = sen_mod_acc / sen_mod
    c_recall = sen_mod_acc / sen_tar_mod
    c_F1 = 2 * c_precision * c_recall / (c_precision + c_recall)
    print("detection token accuracy:{0},precision:{1},recall:{2},F1:{3}".format(d_sen_acc / setsum, d_precision,
                                                                                d_recall, d_F1))
    print("correction token accuracy:{0},precision:{1},recall:{2},F1:{3}".format(sen_acc / setsum,
                                                                                 sen_mod_acc / sen_mod,
                                                                                 sen_mod_acc / sen_tar_mod,
                                                                                 c_F1))
    print("token target modify:{0},token sum:{1},token modified accurate:{2}".format(sen_tar_mod, setsum,
                                                                                     sen_mod_acc))
    # accuracy, precision, recall, F1
    return sen_acc / setsum, sen_mod_acc / sen_mod, sen_mod_acc / sen_tar_mod, c_F1


def remove_space(src_text):
    src_text = re.sub("\s+", "", src_text)
    return src_text


def find_ori_data(path):
    """
    获取原始数据，包括gold
    :param path:
    :return:
    """
    all_text_ori = []
    all_text_trg = []
    with open(path, encoding="utf-8") as f1:
        for line in f1.readlines():
            src, trg = line.strip().split()
            all_text_ori.append(remove_space(src))
            all_text_trg.append(remove_space(trg))
    return all_text_ori, all_text_trg


def get_model_output(model_out_path, all_text_ori):
    """
    :param path:
    :return:
    """
    all_model_outputs = []
    with open(model_out_path, encoding="utf-8") as f:
        all_predicts = f.readlines()
        for s in range(len(all_predicts)):
            src_text = all_text_ori[s]
            line = all_predicts[s]
            key, tmp_text = line.strip().split("[CLS]")
            trg_pre = tmp_text.replace("[SEP]", "").replace("##", "").replace("[UNK]", "N")

            trg_text = ""
            i = 1
            if " " in trg_pre or "##" in trg_pre:
                print("=========")
            # trg_pre = remove_space(trg_pre)
            num = len(src_text)
            for item in list(trg_pre.strip()):
                if item in ["N"]:
                    item = list(src_text)[i - 1]
                trg_text += item
                if i == num:
                    break
                i += 1
            all_model_outputs.append(trg_text)
    return all_model_outputs


if __name__ == "__main__":
    # # 原始数据
    print("===========sighan13 testing====================：")
    data = 13
    path_13 = "/data_local/TwoWaysToImproveCSC/BERT/data/13test_lower.txt"
    all_srcs, all_trgs = find_ori_data(path_13)

    # path_mita = "/data_local/TwoWaysToImproveCSC/BERT/data/13test_lower_mita.txt"
    # _, all_pres = find_ori_data(path_mita)

    path_mita1 = "/data_local/TwoWaysToImproveCSC/BERT/data/13test_lower_mita_punct.txt"
    _, all_pres1 = find_ori_data(path_mita1)

    path_model = "/data_local/TwoWaysToImproveCSC/BERT/data/13test_lower_model.txt"
    _, all_pres_model = find_ori_data(path_model)

    # model_out_path = "/data_local/TwoWaysToImproveCSC/BERT/data_analysis/self_base_998_13test_lower.txt_cor.txt"
    # all_pres = get_model_output(model_out_path, all_srcs)

    print("sighan13 句子级别:")
    sent_mertic(all_srcs, all_pres_model, all_trgs)
    with open("../data/sighan13.out", "w", encoding="utf-8") as fw:
        for src, pre, trg in zip(all_srcs, all_pres_model, all_trgs):
            fw.write(src + " " + pre + " " + trg + "\n")

    sent_mertic(all_srcs, all_pres1, all_trgs)

    print("sighan13 token级别:")
    token_mertic(all_srcs, all_pres_model, all_trgs)
    token_mertic(all_srcs, all_pres1, all_trgs)
    print("===========cc testing====================：")

    path_4 = "/data_local/TwoWaysToImproveCSC/BERT/cc_data/chinese_spell_lower_4.txt"
    all_srcs, all_trgs = find_ori_data(path_4)

    # model_out_path = "/data_local/TwoWaysToImproveCSC/BERT/data_analysis/self_base_998_chinese_spell_lower_4.txt_cor.txt"
    # all_pres = get_model_output(model_out_path, all_srcs)

    # path_mita = "/data_local/TwoWaysToImproveCSC/BERT/data/chinese_spell_lower_4_mita.txt"
    # _, all_pres = find_ori_data(path_mita)

    path_mita1 = "/data_local/TwoWaysToImproveCSC/BERT/data/chinese_spell_lower_4_mita_punct.txt"
    _, all_pres1 = find_ori_data(path_mita1)

    path_model = "/data_local/TwoWaysToImproveCSC/BERT/data/chinese_spell_lower_4_model.txt"
    _, all_pres_model = find_ori_data(path_model)

    print("cc句子级别:")
    sent_mertic(all_srcs, all_pres_model, all_trgs)
    sent_mertic(all_srcs, all_pres1, all_trgs)

    print("cc token级别:")
    token_mertic(all_srcs, all_pres_model, all_trgs)
    token_mertic(all_srcs, all_pres1, all_trgs)

    # print("=后处理=")
    #
    # all_pres_1 = []
    # with open("/data_local/TwoWaysToImproveCSC/BERT/chinese-xinhua/bert_out_cc.pre", "r", encoding="utf-8") as f:
    #     for line in f.readlines():
    #         src, trg = line.strip().split(" ")
    #         all_pres_1.append(trg)
    # print("句子级别：不做预处理")
    # sent_mertic(all_srcs, all_pres_1, all_trgs)
    #
    # print("token级别：不做预处理")
    # token_mertic(all_srcs, all_pres_1, all_trgs)
