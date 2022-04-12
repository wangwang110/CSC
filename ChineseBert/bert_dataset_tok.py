#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
from typing import List

from pypinyin import pinyin, Style
from tokenizers import BertWordPieceTokenizer


class BertMaskDataset(object):

    def __init__(self, vocab_file, config_path, max_length: int = 512):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = BertWordPieceTokenizer(vocab_file)

        # load pinyin map dict
        with open(os.path.join(config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map tensor
        with open(os.path.join(config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)  # 汉字对应的拼音,包括声调
        # load pinyin map tensor
        with open(os.path.join(config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

    def convert_sentence_to_pinyin_ids(self, sentence) -> List[List[int]]:
        """
        self.tokenizer.convert_sentence_to_pinyin_ids(src, tokenizer_output)
        :param sentence:
        :param tokenizer_output: 如何用起来的
        :return:
        """
        pinyin_ids = []
        # get pinyin of a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                pinyin_ids.append([0] * 8)
            elif pinyin_string in self.pinyin2tensor:
                pinyin_ids.append(self.pinyin2tensor[pinyin_string])
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_ids.append(ids)
        return pinyin_ids


if __name__ == '__main__':
    vocab_file = "/data/nfsdata2/sunzijun/glyce/glyce/bert_chinese_base_large_vocab/vocab.txt"
    config_path = "/data/nfsdata2/sunzijun/glyce/glyce/config"
    sentence = "我喜欢猫。"
    tokenizer = BertMaskDataset(vocab_file, config_path)
    input_ids, pinyin_ids = tokenizer.mask_sentence(sentence, 1)
    print(pinyin_ids)
    print(input_ids)
