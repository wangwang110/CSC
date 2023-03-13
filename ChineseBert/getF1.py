#
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import re


def sent_mertic_cor(all_srcs, all_pres, all_trgs):
    """
    句子级别纠正指标：所有位置纠正对才算对
    :param all_pres:
    :param all_trgs:
    :return:
    """

    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    change_num = 0
    for src, tgt_pred, tgt in zip(all_srcs, all_pres, all_trgs):

        if src != tgt_pred:
            change_num += 1

        # 负样本
        if src == tgt:
            # 预测也为负
            if tgt == tgt_pred:
                TN += 1
            # 预测为正
            else:
                FP += 1
                # print('wrong')
        # 正样本
        else:
            # 预测也为正
            if tgt == tgt_pred:
                TP += 1
            # 预测为负
            else:
                FN += 1
        total_num += 1
    acc = (TP + TN) / total_num
    # precision = TP / (TP + FP) if TP > 0 else 0.0
    # 官方评测以及pycorrect计算p值方法，分母忽略了原句被修改，但是没改对的情况，因此计算出的指标偏高
    precision = TP / change_num if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    print(
        f'Sentence Level correction: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}')
    return acc, precision, recall, f1


def sent_mertic_det(all_srcs, all_pres, all_trgs):
    """
    句子级别检测指标：所有位置都检测对才算对
    :param all_pres:
    :param all_trgs:
    :return:
    """

    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    change_num = 0
    for src, tgt_pred, tgt in zip(all_srcs, all_pres, all_trgs):
        src_tgt_tag = [1 if s == t else 0 for s, t in zip(list(src), list(tgt))]
        src_tgt_pred_tag = [1 if s == t else 0 for s, t in zip(list(src), list(tgt_pred))]

        if src != tgt_pred:
            change_num += 1

        # 负样本
        if src == tgt:
            # 预测也为负
            if src == tgt_pred:
                TN += 1
                # print('right')
            # 预测为正
            else:
                FP += 1
                # print('wrong')
        # 正样本
        else:
            # 预测也为正
            if src_tgt_tag == src_tgt_pred_tag:
                TP += 1
                # print('right')
            # 预测为负
            else:
                FN += 1
                # print('wrong')
        total_num += 1
    acc = (TP + TN) / total_num
    # precision = TP / (TP + FP) if TP > 0 else 0.0
    precision = TP / change_num if TP > 0 else 0.0
    # 官方评测以及pycorrect计算p值方法，分母忽略了原句被修改，但是没完全检测对的情况，因此计算出的指标偏高
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    print(
        f'Sentence Level detection: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}')
    return acc, precision, recall, f1
