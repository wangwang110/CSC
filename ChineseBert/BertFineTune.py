import torch.nn as nn


class BertFineTuneMac(nn.Module):
    def __init__(self, bert, device):
        super(BertFineTuneMac, self).__init__()
        self.device = device
        self.config = bert.config
        self.bert = bert.to(device)

    def forward(self, input_ids, input_tyi, input_attn_mask, text_labels, input_pinyin):
        if text_labels is not None:
            text_labels[text_labels == 0] = -100
        else:
            text_labels = None
        bert_outputs = self.bert.forward(input_ids=input_ids, pinyin_ids=input_pinyin, token_type_ids=input_tyi,
                                         attention_mask=input_attn_mask,
                                         labels=text_labels, return_dict=True, output_hidden_states=True)
        #
        # chinese_bert.forward(input_ids, pinyin_ids)[0]
        #
        # self,
        # input_ids = None,
        # pinyin_ids = None,
        # attention_mask = None,
        # token_type_ids = None,
        # position_ids = None,
        # head_mask = None,
        # inputs_embeds = None,
        # encoder_hidden_states = None,
        # encoder_attention_mask = None,
        # labels = None,
        # output_attentions = None,
        # output_hidden_states = None,
        # return_dict = None,
        # ** kwargs

        # prob = self.detection(bert_outputs.hidden_states[-1])
        if text_labels is None:
            outputs = (bert_outputs.logits,)
        else:
            outputs = (
                bert_outputs.logits,
                bert_outputs.loss,)

        return outputs
