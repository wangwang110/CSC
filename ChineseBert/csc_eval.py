# -*- coding: UTF-8 -*-


import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import construct, BertDataset, li_testconstruct
from BertFineTune import BertFineTuneMac
# from datasets.bert_dataset_tok import BertMaskDataset
from datasets.bert_mask_dataset import BertMaskDataset
from models.modeling_glycebert import GlyceBertForMaskedLM


class CSCmodel:
    def __init__(self, chinese_bert_path, model_path, gpu_id="6"):
        """
        :param bert_path:
        :param model_path:
        :param gpu_id:
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型加载
        chinese_bert = GlyceBertForMaskedLM.from_pretrained(chinese_bert_path, return_dict=True)
        vocab_file = os.path.join(chinese_bert_path, 'vocab.txt')
        config_path = os.path.join(chinese_bert_path, 'config')
        self.tokenizer = BertMaskDataset(vocab_file, config_path)

        self.model = BertFineTuneMac(chinese_bert, self.device).to(self.device)
        self.model.load_state_dict(torch.load(model_path))

        self.batch_size = 20

        # bert的词典
        self.vob = {}
        with open("/data_local/plm_models/chinese_L-12_H-768_A-12/vocab.txt", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.vob.setdefault(i, line.strip())

    def test_without_trg(self, all_texts):
        self.model.eval()
        test = li_testconstruct(all_texts)
        test = BertDataset(self.tokenizer, test)
        test = DataLoader(test, batch_size=int(self.batch_size), shuffle=False)
        pres = []
        srcs = []
        for batch in test:
            inputs = self.help_vectorize(batch)
            max_len = 180
            input_ids, input_tyi, input_attn_mask, input_pinyin = inputs['input_ids'][:, :max_len], \
                                                                  inputs['token_type_ids'][:, :max_len], \
                                                                  inputs['attention_mask'][:, :max_len], \
                                                                  inputs['pinyin_ids'][:, :max_len, :]
            output_ids = None
            outputs = self.model(input_ids, input_tyi, input_attn_mask, output_ids, input_pinyin)
            out = outputs[0].argmax(dim=-1)
            num = len(batch["input"])
            for i in range(num):
                src = batch["input"][i]
                srcs.append(src)
                tokens = list(src)
                for j in range(len(tokens) + 1):
                    if out[i][j + 1] != input_ids[i][j + 1] and out[i][j + 1] not in [100, 101, 102, 0]:
                        val = out[i][j + 1].item()
                        if j < len(tokens):
                            tokens[j] = self.vob[val]
                out_sent = "".join(tokens)
                pres.append(out_sent)
        return srcs, pres

    def help_vectorize(self, batch):
        """
        :param batch:
        :return:
        """
        src_li, trg_li = batch['input'], batch['output']
        max_seq_length = max([len(src) for src in src_li]) + 2
        inputs = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'pinyin_ids': []}

        for src, trg in zip(src_li, trg_li):
            input_ids, input_mask, segment_ids, pinyin_tokens = self.text2vec(src, max_seq_length)
            inputs['input_ids'].append(input_ids)
            inputs['token_type_ids'].append(segment_ids)
            inputs['attention_mask'].append(input_mask)
            inputs['pinyin_ids'].append(pinyin_tokens)

        inputs['input_ids'] = torch.tensor(np.array(inputs['input_ids'])).to(self.device)
        inputs['token_type_ids'] = torch.tensor(np.array(inputs['token_type_ids'])).to(self.device)
        inputs['attention_mask'] = torch.tensor(np.array(inputs['attention_mask'])).to(self.device)
        inputs['pinyin_ids'] = torch.tensor(np.array(inputs['pinyin_ids'])).to(self.device)
        return inputs

    def text2vec(self, src, max_seq_length):
        """
        :param src:
        :return:
        """
        # convert sentence to ids
        tokenizer_output = self.tokenizer.tokenizer.encode(src)
        input_ids = tokenizer_output.ids
        input_mask = tokenizer_output.attention_mask
        segment_ids = tokenizer_output.type_ids

        pinyin_tokens = self.tokenizer.convert_sentence_to_pinyin_ids(src, tokenizer_output)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            pinyin_tokens.append([0] * 8)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        return input_ids, input_mask, segment_ids, pinyin_tokens


if __name__ == "__main__":
    # 初始化模型
    chinese_bert_path = "/data_local/plm_models/ChineseBERT-base/"
    load_path = "/data_local/ChineseBert/save/wang2018/sighan13/model.pkl"
    obj = CSCmodel(chinese_bert_path, load_path)
    input_text, model_ouput = obj.test_without_trg([
        "布告栏转眼之间从不起眼的丑小鸭变成了高贵优雅的天鹅！仅管这大改造没有得名，但过程也是很可贵的。",
        "我爱北进天安门",
        "我爱北京天按门",
        "没过几分钟，救护车来了，发出响亮而清翠的声音",
        "我见过一望无际、波澜壮阔的大海；玩赏过水平如镜、诗情画意的西湖；游览过翡翠般的漓江；让我难以忘怀的要数那荷叶飘香、群山坏绕的普者黑。"])
    for src, trg in zip(input_text, model_ouput):
        print(src)
        print(trg)
        for s, t in zip(list(src), list(trg)):
            if s != t:
                print(s + "->" + t)
