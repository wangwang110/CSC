


# CSC
中文拼写纠错

本项目只关注Chinese Spelling Check (CSC)，不考虑多字少字的语法纠错。

关于语法纠错可参考 https://github.com/HillZhang1999/MuCGEC 


常用数据来源:

SIGHAN Bake-off 2013: http://ir.itc.ntnu.edu.tw/lre/sighan7csc.html

SIGHAN Bake-off 2014: http://ir.itc.ntnu.edu.tw/lre/clp14csc.html

SIGHAN Bake-off 2015: http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html

Wang271K: https://github.com/wdimmy/Automatic-Corpus-Generation/tree/master/corpus 

sighan13，sighan14，sighan15 包含对应的训练集和测试集，Wang271K是论文利用OCR以及语音识别的方法生成的数据构，仅仅用来训练



## 评价指标

使用句子级纠正F1值。

关于句子级别纠正P值，有两种计算方式
1. 分母不考虑修改了原句但是与正确句子不同的情况
2. 分母考虑修改了原句但是与正确句子不同的情况

使用第一种计算方式得到的结果偏高，这里采用第2种

[代码](https://github.com/wangwang110/ChineseBert_CSC/blob/main/ChineseBert/getF1.py)

该指标更为严格（所有修正过的句子都算作P值的分母），论文大都使用该评价指标


## 榜单

各论文指标（不同的论文使用bert取得的结果不一样,可能是使用的预训练数据不一样或者超参数的设置不同）


sighan13结果:

 模型 | Detection Level | Correction Level | 来源
---|---|--- |---
 Hybrid | 54.0 69.3 60.7 |  _ _ 52.1 | [A Hybrid Approach to Automatic Corpus Generation for Chinese Spelling Check](https://aclanthology.org/D18-1273.pdf)
 FASpell |76.2 63.2 69.1| 73.1 60.5 66.2 | [FASPell: A Fast, Adaptable, Simple, Powerful Chinese Spell Checker Based On DAE-Decoder Paradigm](https://aclanthology.org/D19-5522.pdf)
 Google BERT | 79.0 72.8 75.8 | 77.7 71.6 74.6 |  [SpellGCN: Incorporating Phonological and Visual Similarities into Language Models for Chinese Spelling Check](https://arxiv.org/pdf/2004.14166.pdf)
 SpellGCN | 80.1 74.4 77.2 | 78.3 72.7 75.4 | [SpellGCN: Incorporating Phonological and Visual Similarities into Language Models for Chinese Spelling Check](https://arxiv.org/pdf/2004.14166.pdf)
 HeadFilt |100.0 74.9 85.7| 100.0 74.1 85.1 | [Domain-shift Conditioning using Adaptable Filtering via Hierarchical Embeddings for Robust Chinese Spell Check](https://arxiv.org/pdf/2008.12281.pdf) 
 Chunk-based CSC |61.19 75.67 67.66| 74.34 67.20 70.59 |  [Chunk-based Chinese Spelling Check with Global Optimization](https://aclanthology.org/2020.findings-emnlp.184.pdf) 
 BERT + Pre-trained for CSC  | - - 84.9 | - - 84.4 | [Exploration and Exploitation: Two Ways to Improve Chinese Spelling Correction Models](https://aclanthology.org/2021.acl-short.56.pdf)
 BERT + Adversarial training  | - - 84.0 | - - 83.5 | [Exploration and Exploitation: Two Ways to Improve Chinese Spelling Correction Models](https://aclanthology.org/2021.acl-short.56.pdf)
 SpellGCN+ | 85.7 78.8 82.1 | 84.6 77.8 81.0 | [Read, Listen, and See: Leveraging Multimodal Information Helps Chinese Spell Checking](https://aclanthology.org/2021.findings-acl.64.pdf)
 BERT-wwm  | 85.0 77.0 80.8| 83.0 75.2 78.9 | [Read, Listen, and See: Leveraging Multimodal Information Helps Chinese Spell Checking](https://aclanthology.org/2021.findings-acl.64.pdf)
 REALISE (BERT-wwm ) | 88.6 82.5 85.4 | 887.2 81.2 84.1 | [Read, Listen, and See: Leveraging Multimodal Information Helps Chinese Spell Checking](https://aclanthology.org/2021.findings-acl.64.pdf)
 BERT_CRS |84.8 79.5 82.1 | 83.9 78.7 81.2| [Global Attention Decoder for Chinese Spelling Error Correction](https://aclanthology.org/2021.findings-acl.122.pdf)
 BERT_CRS_GAD | 85.7 79.5 82.5 | 84.9 78.7 81.6|[Global Attention Decoder for Chinese Spelling Error Correction](https://aclanthology.org/2021.findings-acl.122.pdf)
 ERNIE  |76.6 71.9 74.2 | 73.0 68.5 70.6 | [Correcting Chinese Spelling Errors with Phonetic Pre-training](https://aclanthology.org/2021.findings-acl.198.pdf)
 MLM-phonetics | 82.0 78.3 80.1| 79.5 77.0 78.2 | [Correcting Chinese Spelling Errors with Phonetic Pre-training](https://aclanthology.org/2021.findings-acl.198.pdf)
 RoBERTa  |85.4 77.7 81.3 | 83.9 76.4 79.9 | [Dynamic Connected Networks for Chinese Spelling Check](https://aclanthology.org/2021.findings-acl.216.pdf)
 RoBERTa-DCN | 86.2 78.4 82.1| 84.6 76.9 80.5 | [Dynamic Connected Networks for Chinese Spelling Check](https://aclanthology.org/2021.findings-acl.216.pdf)
 RoBERTa-Pretrain-DCN | 86.8 79.6 83.0| 84.7 77.7 81.0 | [Dynamic Connected Networks for Chinese Spelling Check](https://aclanthology.org/2021.findings-acl.216.pdf)
 Google BERT  |98.7 70.6 82.3|98.6 67.8 80.4| [PHMOSpell: Phonological and Morphological Knowledge Guided Chinese Spelling Check](https://aclanthology.org/2021.acl-long.464.pdf)
 PHMOSpell |99.5 76.8 86.7| 99.5 75.1 85.6 | [PHMOSpell: Phonological and Morphological Knowledge Guided Chinese Spelling Check](https://aclanthology.org/2021.acl-long.464.pdf)
 ECOPO BERT-wwm  |87.2 81.7 84.4 | 86.1 80.6 83.3|  [The Past Mistake is the Future Wisdom: Error-driven Contrastive Probability Optimization for Chinese Spell Checking](https://arxiv.org/pdf/2203.00991.pdf) 
 ECOPO REALISE (BERT-wwm )|89.3 83.2 86.2 | 88.5 82.0 85.1| [The Past Mistake is the Future Wisdom: Error-driven Contrastive Probability Optimization for Chinese Spell Checking](https://arxiv.org/pdf/2203.00991.pdf)
 Google BERT |79.0 72.8 75.8| 77.7 71.6 74.6 |  [MDCSpell: A Multi-task Detector-Corrector Framework for Chinese Spelling Correction](https://aclanthology.org/2022.findings-acl.98.pdf) 
 MDCSpell |89.1 78.3 83.4| 87.5 76.8 81.8| [MDCSpell: A Multi-task Detector-Corrector Framework for Chinese Spelling Correction](https://aclanthology.org/2022.findings-acl.98.pdf)
 Soft-Masked BERT | 81.1 75.7 78.3 |  75.1 70.1 72.5|  [Contextual Similarity is More Valuable than Character Similarity: An Empirical Study for Chinese Spell Checking](https://arxiv.org/pdf/2207.09217.pdf) 
 CL (Soft-Masked BERT) |84.7 77.0 80.7| 80.9 74.5 77.6| [Contextual Similarity is More Valuable than Character Similarity: An Empirical Study for Chinese Spell Checking](https://arxiv.org/pdf/2207.09217.pdf)
 Google BERT | 98.7 70.6 82.3| 98.6 67.8 80.4 |  [Contextual Similarity is More Valuable than Character Similarity: An Empirical Study for Chinese Spell Checking](https://arxiv.org/pdf/2207.09217.pdf) 
 CL (BERT) |99.1 74.8 85.3| 99.1 73.2 84.2| [Contextual Similarity is More Valuable than Character Similarity: An Empirical Study for Chinese Spell Checking](https://arxiv.org/pdf/2207.09217.pdf)
 MacBERT |98.7 70.8 82.5|  98.6 67.9 80.4|  [Contextual Similarity is More Valuable than Character Similarity: An Empirical Study for Chinese Spell Checking](https://arxiv.org/pdf/2207.09217.pdf) 
 CL (MacBERT) |99.3 75.7 85.9| 99.2 73.8 84.6| [Contextual Similarity is More Valuable than Character Similarity: An Empirical Study for Chinese Spell Checking](https://arxiv.org/pdf/2207.09217.pdf)
 uChecker | 75.4 73.4 74.4 | 72.6 70.8 71.7|  [uChecker: Masked Pretrained Language Models as Unsupervised Chinese Spelling Checkers](https://arxiv.org/pdf/2209.07068.pdf) 
 Google BERT  |85.0 77.0 80.8| 83.0 75.2 78.9|  [Learning from the Dictionary: Heterogeneous Knowledge Guided Fine-tuning for Chinese Spell Checking](https://arxiv.org/pdf/2210.10320.pdf) 
 LEAD | 88.3 83.4 85.8| 87.2 82.4 84.7|  [Learning from the Dictionary: Heterogeneous Knowledge Guided Fine-tuning for Chinese Spell Checking](https://arxiv.org/pdf/2210.10320.pdf) 
 SCOPE(ChineseBert) |87.4 83.4 85.4|  86.3 82.4 84.3|  [Improving Chinese Spelling Check by Character Pronunciation Prediction: The Effects of Adaptivity and Granularity](https://arxiv.org/pdf/2210.10996.pdf) 
 SCOPE(REALISE) |87.5 83.2 85.3|86.4 82.3 84.3|  [Improving Chinese Spelling Check by Character Pronunciation Prediction: The Effects of Adaptivity and Granularity](https://arxiv.org/pdf/2210.10996.pdf) 
 SDCL | 88.9 81.8 85.2| 88.0 81.0 84.3|  [SDCL: Self-Distillation Contrastive Learning for Chinese Spell Checking](https://arxiv.org/pdf/2210.17168.pdf) 
 InfoKNN-CSC | 89.7 82.8 86.1 | 82.1 88.7 81.9 85.2|  [Chinese Spelling Check with Nearest Neighbors](https://arxiv.org/pdf/2211.07843.pdf) 

sighan14结果:

 模型 | Detection Level | Correction Level |来源
---|---|--- |---
 Hybrid | 51.9 66.2 58.2 | _ _ 56.1 | [A Hybrid Approach to Automatic Corpus Generation for Chinese Spelling Check](https://aclanthology.org/D18-1273.pdf)
 FASpell |61.0 53.5 57.0| 59.4 52.0 55.4 | [FASPell: A Fast, Adaptable, Simple, Powerful Chinese Spell Checker Based On DAE-Decoder Paradigm](https://aclanthology.org/D19-5522.pdf)
 Google BERT | 65.6 68.1 66.8 | 63.1 65.5 64.3 | [SpellGCN: Incorporating Phonological and Visual Similarities into Language Models for Chinese Spelling Check](https://arxiv.org/pdf/2004.14166.pdf)
 SpellGCN | 65.1 69.5 67.2 | 63.1 67.2 65.3 | [SpellGCN: Incorporating Phonological and Visual Similarities into Language Models for Chinese Spelling Check](https://arxiv.org/pdf/2004.14166.pdf)
 HeadFilt |82.5 61.6 70.5| 82.1 60.2 69.4 |  [Domain-shift Conditioning using Adaptable Filtering via Hierarchical Embeddings for Robust Chinese Spell Check](https://arxiv.org/pdf/2008.12281.pdf) 
 Chunk-based CSC | 78.65 54.80 64.59| 77.43 51.04 61.52 |  [Chunk-based Chinese Spelling Check with Global Optimization](https://aclanthology.org/2020.findings-emnlp.184.pdf) 
 BERT + Pre-trained for CSC  | - - 70.4 | - - 68.6 | [Exploration and Exploitation: Two Ways to Improve Chinese Spelling Correction Models](https://aclanthology.org/2021.acl-short.56.pdf)
 BERT + Adversarial training  | - - 68.4 | - - 66.8 | [Exploration and Exploitation: Two Ways to Improve Chinese Spelling Correction Models](https://aclanthology.org/2021.acl-short.56.pdf)
 BERT-wwm  | 64.5 68.6 66.5| 62.4 66.3 64.3| [Read, Listen, and See: Leveraging Multimodal Information Helps Chinese Spell Checking](https://aclanthology.org/2021.findings-acl.64.pdf)
 REALISE (BERT-wwm )| 67.8 71.5 69.6 | 66.3 70.0 68.1 | [Read, Listen, and See: Leveraging Multimodal Information Helps Chinese Spell Checking](https://aclanthology.org/2021.findings-acl.64.pdf)
 BERT_CRS  |65.4 72.7 68.9 | 63.4 70.4 66.7| [Global Attention Decoder for Chinese Spelling Error Correction](https://aclanthology.org/2021.findings-acl.122.pdf)
 BERT_CRS_GAD | 66.6 71.8 69.1| 65.0 70.1 67.5|[Global Attention Decoder for Chinese Spelling Error Correction](https://aclanthology.org/2021.findings-acl.122.pdf)
 ERNIE  |63.5 69.3 66.3 | 60.1 65.6 62.8| [Correcting Chinese Spelling Errors with Phonetic Pre-training](https://aclanthology.org/2021.findings-acl.198.pdf)
 MLM-phonetics | 66.2 73.8 69.8| 64.2 73.8 68.7 | [Correcting Chinese Spelling Errors with Phonetic Pre-training](https://aclanthology.org/2021.findings-acl.198.pdf)
 RoBERTa  |64.2 68.4 66.2 | 62.7 66.7 64.6 | [Dynamic Connected Networks for Chinese Spelling Check](https://aclanthology.org/2021.findings-acl.216.pdf)
 RoBERTa-DCN | 67.6 68.6 68.0| 64.9 65.9 65.4 | [Dynamic Connected Networks for Chinese Spelling Check](https://aclanthology.org/2021.findings-acl.216.pdf)
 RoBERTa-Pretrain-DCN | 67.4 70.4 68.9| 65.8 68.7 67.2 | [Dynamic Connected Networks for Chinese Spelling Check](https://aclanthology.org/2021.findings-acl.216.pdf)
 Google BERT  |78.6 60.7 68.5|77.8 57.6 66.2| [PHMOSpell: Phonological and Morphological Knowledge Guided Chinese Spelling Check](https://aclanthology.org/2021.acl-long.464.pdf)
 PHMOSpell |85.3 67.6 75.5| 84.7 64.3 73.1 |[PHMOSpell: Phonological and Morphological Knowledge Guided Chinese Spelling Check](https://aclanthology.org/2021.acl-long.464.pdf)
 Google BERT(4 layer) |82.6 59.0 68.8| 82.4 58.0 68.1| [SpellBERT: A Lightweight Pretrained Model for Chinese Spelling Check](https://aclanthology.org/2021.emnlp-main.287v2.pdf)
 SpellBERT (4 layer) |83.1 62.0 71.0| 82.9 61.2 70.4| [SpellBERT: A Lightweight Pretrained Model for Chinese Spelling Check](https://aclanthology.org/2021.emnlp-main.287v2.pdf)
 ECOPO BERT-wwm  |65.8 69.0 67.4 | 63.7 66.9 65.3|  [The Past Mistake is the Future Wisdom: Error-driven Contrastive Probability Optimization for Chinese Spell Checking](https://arxiv.org/pdf/2203.00991.pdf) 
 ECOPO REALISE (BERT-wwm ) |68.8 72.1 70.4 | 67.5 71.0 69.2| [The Past Mistake is the Future Wisdom: Error-driven Contrastive Probability Optimization for Chinese Spell Checking](https://arxiv.org/pdf/2203.00991.pdf)
 Google BERT |65.6 68.1 66.8| 63.1 65.5 64.3 |  [MDCSpell: A Multi-task Detector-Corrector Framework for Chinese Spelling Correction](https://aclanthology.org/2022.findings-acl.98.pdf) 
 MDCSpell |70.2 68.8 69.5| 69.0 67.7 68.3| [MDCSpell: A Multi-task Detector-Corrector Framework for Chinese Spelling Correction](https://aclanthology.org/2022.findings-acl.98.pdf)
 Soft-Masked BERT | 65.2 70.4 67.7|  63.7 68.7 66.1|  [Contextual Similarity is More Valuable than Character Similarity: An Empirical Study for Chinese Spell Checking](https://arxiv.org/pdf/2207.09217.pdf) 
 CL (Soft-Masked BERT) |68.4 70.9 69.6| 67.8 69.1 68.4| [Contextual Similarity is More Valuable than Character Similarity: An Empirical Study for Chinese Spell Checking](https://arxiv.org/pdf/2207.09217.pdf)
 Google BERT | 78.6 60.7 68.5| 77.8 57.6 66.2 |  [Contextual Similarity is More Valuable than Character Similarity: An Empirical Study for Chinese Spell Checking](https://arxiv.org/pdf/2207.09217.pdf) 
 CL (BERT) |79.2 61.6 69.3| 78.5 60.8 68.5| [Contextual Similarity is More Valuable than Character Similarity: An Empirical Study for Chinese Spell Checking](https://arxiv.org/pdf/2207.09217.pdf)
 MacBERT |78.8 61.0 68.8|  78.0 58.0 66.5|  [Contextual Similarity is More Valuable than Character Similarity: An Empirical Study for Chinese Spell Checking](https://arxiv.org/pdf/2207.09217.pdf) 
 CL (MacBERT) |79.7 62.4 70.0| 79.0 61.4 69.1| [Contextual Similarity is More Valuable than Character Similarity: An Empirical Study for Chinese Spell Checking](https://arxiv.org/pdf/2207.09217.pdf)
 uChecker | 61.7 61.5 61.6 |  57.6 57.5 57.6|  [uChecker: Masked Pretrained Language Models as Unsupervised Chinese Spelling Checkers](https://arxiv.org/pdf/2209.07068.pdf) 
 Google BERT  |64.5 68.6 66.5|  62.4 66.3 64.3|  [Learning from the Dictionary: Heterogeneous Knowledge Guided Fine-tuning for Chinese Spell Checking](https://arxiv.org/pdf/2210.10320.pdf) 
 LEAD | 70.7 71.0 70.8| 69.3 69.6 69.5|  [Learning from the Dictionary: Heterogeneous Knowledge Guided Fine-tuning for Chinese Spell Checking](https://arxiv.org/pdf/2210.10320.pdf) 
 SCOPE(ChineseBert) |70.1 73.1 71.6|   68.6 71.5 70.1|  [Improving Chinese Spelling Check by Character Pronunciation Prediction: The Effects of Adaptivity and Granularity](https://arxiv.org/pdf/2210.10996.pdf) 
 SCOPE(REALISE) |69.0 75.0 71.9| 67.1 72.9 69.9|  [Improving Chinese Spelling Check by Character Pronunciation Prediction: The Effects of Adaptivity and Granularity](https://arxiv.org/pdf/2210.10996.pdf) 
 SDCL | 69.7 70.3 70.0| 70.2 67.5 68.8|  [SDCL: Self-Distillation Contrastive Learning for Chinese Spell Checking](https://arxiv.org/pdf/2210.17168.pdf) 
 InfoKNN-CSC | 72.1 70.6 71.3 | 71.3 69.8 70.6|  [Chinese Spelling Check with Nearest Neighbors](https://arxiv.org/pdf/2211.07843.pdf) 

sighan15结果:

 模型 | Detection Level | Correction Level | 来源
---|---|---|---
 Hybrid | 56.6 69.4 62.3 |  _ _ 57.1  | [A Hybrid Approach to Automatic Corpus Generation for Chinese Spelling Check](https://aclanthology.org/D18-1273.pdf)
 FASpell | 67.6 60.0 63.5 | 66.6 59.1 62.6 | [FASPell: A Fast, Adaptable, Simple, Powerful Chinese Spell Checker Based On DAE-Decoder Paradigm](https://aclanthology.org/D19-5522.pdf)
 Google BERT | 73.7 78.2 75.9 | 70.9 75.2 73.0 | [SpellGCN: Incorporating Phonological and Visual Similarities into Language Models for Chinese Spelling Check](https://arxiv.org/pdf/2004.14166.pdf)
 SpellGCN | 74.8 80.7 77.7 | 72.1 77.7 75.9 | [SpellGCN: Incorporating Phonological and Visual Similarities into Language Models for Chinese Spelling Check](https://arxiv.org/pdf/2004.14166.pdf)
 Google BERT | 73.0 70.8 71.9 | 65.9 64.0 64.9 | [Spelling Error Correction with Soft-Masked BERT](https://arxiv.org/pdf/2005.07421.pdf)
 Soft-Masked BERT | 73.7 73.2 73.5 | 66.7 66.2 66.4 | [Spelling Error Correction with Soft-Masked BERT](https://arxiv.org/pdf/2005.07421.pdf)
 HeadFilt |84.5 71.8 77.6| 84.2 70.2 76.5| [Domain-shift Conditioning using Adaptable Filtering via Hierarchical Embeddings for Robust Chinese Spell Check](https://arxiv.org/pdf/2008.12281.pdf) 
 Chunk-based CSC | 88.11 62.00 72.79| 87.33 57.64 69.44 |  [Chunk-based Chinese Spelling Check with Global Optimization](https://aclanthology.org/2020.findings-emnlp.184.pdf) 
 BERT + Pre-trained for CSC  | - - 79.8 | - - 78.0 | [Exploration and Exploitation: Two Ways to Improve Chinese Spelling Correction Models](https://aclanthology.org/2021.acl-short.56.pdf)
 BERT + Adversarial training  | - - 80.0 | - - 78.2 | [Exploration and Exploitation: Two Ways to Improve Chinese Spelling Correction Models](https://aclanthology.org/2021.acl-short.56.pdf)
 BERT-wwm  | 74.2 78.0 76.1 | 71.6 75.3 73.4 | [Read, Listen, and See: Leveraging Multimodal Information Helps Chinese Spell Checking](https://aclanthology.org/2021.findings-acl.64.pdf)
 REALISE (BERT-wwm)  | 77.3 81.3 79.3 | 75.9 79.9 77.8 | [Read, Listen, and See: Leveraging Multimodal Information Helps Chinese Spell Checking](https://aclanthology.org/2021.findings-acl.64.pdf)
 BERT_CRS  |74.0 80.2 77.2 | 72.2 77.8 74.8| [Global Attention Decoder for Chinese Spelling Error Correction](https://aclanthology.org/2021.findings-acl.122.pdf)
 BERT_CRS_GAD | 75.6 80.4 77.9| 73.2 77.8 75.4 |[Global Attention Decoder for Chinese Spelling Error Correction](https://aclanthology.org/2021.findings-acl.122.pdf)
 ERNIE  |73.6 79.8 76.6 | 68.6 74.4 71.4| [Correcting Chinese Spelling Errors with Phonetic Pre-training](https://aclanthology.org/2021.findings-acl.198.pdf)
 MLM-phonetics | 77.5 83.1 80.2| 74.9 80.2 77.5 |[Correcting Chinese Spelling Errors with Phonetic Pre-training](https://aclanthology.org/2021.findings-acl.198.pdf)
 RoBERTa  |74.7 77.3 76.0 | 72.1 74.5 73.3 | [Dynamic Connected Networks for Chinese Spelling Check](https://aclanthology.org/2021.findings-acl.216.pdf)
 RoBERTa-DCN | 76.6 79.8 78.2| 74.2 77.3 75.7 | [Dynamic Connected Networks for Chinese Spelling Check](https://aclanthology.org/2021.findings-acl.216.pdf)
 RoBERTa-Pretrain-DCN |77.1 80.9 79.0| 74.5 78.2 76.3 | [Dynamic Connected Networks for Chinese Spelling Check](https://aclanthology.org/2021.findings-acl.216.pdf)
 Google BERT  |68.4 77.6 72.7 | 66.0 74.9 70.2| [PLOME: Pre-training with Misspelled Knowledge for Chinese Spelling Correction](https://aclanthology.org/2021.acl-long.233.pdf)
 PLOME | 77.4 81.5 79.4 | 75.3 79.3 77.2 | [PLOME: Pre-training with Misspelled Knowledge for Chinese Spelling Correction](https://aclanthology.org/2021.acl-long.233.pdf)
 Google BERT  |79.9 84.1 72.9 78.1|83.1 68.0 74.8| [PHMOSpell: Phonological and Morphological Knowledge Guided Chinese Spelling Check](https://aclanthology.org/2021.acl-long.464.pdf)
 PHMOSpell |90.1 72.7 80.5| 89.6 69.2 78.1| [PHMOSpell: Phonological and Morphological Knowledge Guided Chinese Spelling Check](https://aclanthology.org/2021.acl-long.464.pdf)
 Soft-Masked BERT SSCL | 86.34 72.46 78.79 | 85.20 65.99 74.38 | [Self-Supervised Curriculum Learning for Spelling Error Correction](https://aclanthology.org/2021.emnlp-main.281.pdf)
 Google BERT(4 layer) |85.2 68.9 76.2| 84.8 66.9 74.8| [SpellBERT: A Lightweight Pretrained Model for Chinese Spelling Check](https://aclanthology.org/2021.emnlp-main.287v2.pdf)
 SpellBERT (4 layer) |87.5 73.6 80.0| 87.1 71.5 78.5| [SpellBERT: A Lightweight Pretrained Model for Chinese Spelling Check](https://aclanthology.org/2021.emnlp-main.287v2.pdf)
 ECOPO BERT-wwm |78.2 82.3 80.2| 76.6 80.4 78.4 |  [The Past Mistake is the Future Wisdom: Error-driven Contrastive Probability Optimization for Chinese Spell Checking](https://arxiv.org/pdf/2203.00991.pdf) 
 ECOPO REALISE (BERT-wwm) |77.5 82.6 80.0| 76.1 81.2 78.5| [The Past Mistake is the Future Wisdom: Error-driven Contrastive Probability Optimization for Chinese Spell Checking](https://arxiv.org/pdf/2203.00991.pdf)
 Google BERT |73.7 78.2 75.9| 70.9 75.2 73.0 |  [MDCSpell: A Multi-task Detector-Corrector Framework for Chinese Spelling Correction](https://aclanthology.org/2022.findings-acl.98.pdf) 
 MDCSpell |80.8 80.6 80.7| 78.4 78.2 78.3| [MDCSpell: A Multi-task Detector-Corrector Framework for Chinese Spelling Correction](https://aclanthology.org/2022.findings-acl.98.pdf)
 Google BERT  | 76.0 81.0 78.4|   74.7 79.5 77.0|  [General and Domain Adaptive Chinese Spelling Check with Error Consistent Pretraining](https://arxiv.org/pdf/2203.10929.pdf) 
 ECSpell |81.1 83.0 81.0|    77.5 81.7 79.5|  [General and Domain Adaptive Chinese Spelling Check with Error Consistent Pretraining](https://arxiv.org/pdf/2203.10929.pdf) 
 Soft-Masked BERT |73.7 73.2 73.5| 66.7 66.2 66.4|  [Contextual Similarity is More Valuable than Character Similarity: An Empirical Study for Chinese Spell Checking](https://arxiv.org/pdf/2207.09217.pdf) 
 CL (Soft-Masked BERT) |83.5 74.8 78.9| 79.9 72.1 75.8| [Contextual Similarity is More Valuable than Character Similarity: An Empirical Study for Chinese Spell Checking](https://arxiv.org/pdf/2207.09217.pdf)
 Google BERT | 84.1 72.9 78.1| 83.1 68.0 74.8 |  [Contextual Similarity is More Valuable than Character Similarity: An Empirical Study for Chinese Spell Checking](https://arxiv.org/pdf/2207.09217.pdf) 
 CL (BERT) |85.0 74.5 79.4| 84.2 72.3 77.8| [Contextual Similarity is More Valuable than Character Similarity: An Empirical Study for Chinese Spell Checking](https://arxiv.org/pdf/2207.09217.pdf)
 MacBERT |84.3 73.1 78.3|  83.3 68.2 75.0|  [Contextual Similarity is More Valuable than Character Similarity: An Empirical Study for Chinese Spell Checking](https://arxiv.org/pdf/2207.09217.pdf) 
 CL (MacBERT) |85.8 75.4 80.3| 84.7 73.0 78.4| [Contextual Similarity is More Valuable than Character Similarity: An Empirical Study for Chinese Spell Checking](https://arxiv.org/pdf/2207.09217.pdf)
 uChecker | 75.4 72.0 73.7 |  70.6 67.3 68.9|  [uChecker: Masked Pretrained Language Models as Unsupervised Chinese Spelling Checkers](https://arxiv.org/pdf/2209.07068.pdf) 
 Google BERT  |74.2 78.0 76.1|    71.6 75.3 73.4|  [Learning from the Dictionary: Heterogeneous Knowledge Guided Fine-tuning for Chinese Spell Checking](https://arxiv.org/pdf/2210.10320.pdf) 
 LEAD | 79.2 82.8 80.9| 77.6 81.2 79.3|  [Learning from the Dictionary: Heterogeneous Knowledge Guided Fine-tuning for Chinese Spell Checking](https://arxiv.org/pdf/2210.10320.pdf) 
 SCOPE(ChineseBert) |81.1 84.3 82.7| 79.2 82.3 80.7|  [Improving Chinese Spelling Check by Character Pronunciation Prediction: The Effects of Adaptivity and Granularity](https://arxiv.org/pdf/2210.10996.pdf) 
 SCOPE(REALISE) |78.7 84.7 81.6 | 76.8 82.6 79.6|  [Improving Chinese Spelling Check by Character Pronunciation Prediction: The Effects of Adaptivity and Granularity](https://arxiv.org/pdf/2210.10996.pdf) 
 SDCL | 81.2 79.1 80.1| 79.3 77.5 78.3|  [SDCL: Self-Distillation Contrastive Learning for Chinese Spell Checking](https://arxiv.org/pdf/2210.17168.pdf) 
 InfoKNN-CSC | 81.2 81.2 81.2| 80.0 80.0 80.0|  [Chinese Spelling Check with Nearest Neighbors](https://arxiv.org/pdf/2211.07843.pdf) 
 Google BERT |74.2 78.0 76.1| 71.6 75.3 73.4|  [An Error-Guided Correction Model for Chinese Spelling Error Correction](https://arxiv.org/pdf/2301.06323.pdf) 
 EGCM |82.7 77.6 80.0|  80.6 74.7 77.5|  [An Error-Guided Correction Model for Chinese Spelling Error Correction](https://arxiv.org/pdf/2301.06323.pdf) 
 Pre-Tn EGCM |83.4 79.8 81.6|    81.4 78.4 79.9|  [An Error-Guided Correction Model for Chinese Spelling Error Correction](https://arxiv.org/pdf/2301.06323.pdf) 
 
 
## 香侬科技ChineseBert用于中文拼写纠错

ChineseBert repo[https://github.com/ShannonAI/ChineseBert]

![image](https://user-images.githubusercontent.com/21475557/164133645-e560f580-4c54-4f38-9bca-153f6bf0fd32.png)

- 训练：

1. 下载ChineseBert放出的预训练模型，放置在本地文件夹（chinese_bert_path 参数）

2. 拷贝ChineseBert代码，置于ChineseBert文件夹，并安装ChineseBert所需依赖

3. 运行train.sh


- 测试：

运行eval.sh


- 纠正文本：

填入模型路径，运行csc_eval.py 即可

运行结果:

```
布告栏转眼之间从不起眼的丑小鸭变成了高贵优雅的天鹅！仅管这大改造没有得名，但过程也是很可贵的。
布告栏转眼之间从不起眼的丑小鸭变成了高贵优雅的天鹅！尽管这大改造没有得名，但过程也是很可贵的。
仅->尽
我爱北进天安门
我爱北进天安门
我爱北京天按门
我爱北京天安门
按->安
没过几分钟，救护车来了，发出响亮而清翠的声音
没过几分钟，救护车来了，发出响亮而清翠的声音
我见过一望无际、波澜壮阔的大海；玩赏过水平如镜、诗情画意的西湖；游览过翡翠般的漓江；让我难以忘怀的要数那荷叶飘香、群山坏绕的普者黑。
我见过一望无际、波澜壮阔的大海；玩赏过水平如镜、诗情画意的西湖；游览过翡翠般的漓江；让我难以忘怀的要数那荷叶飘香、群山环绕的普者黑。
坏->环
```


- 已经训练好的模型：

链接: https://pan.baidu.com/s/1mi0r2Uvv9rd_bfDONNDVYA?pwd=cvak 

- 指标：

数据集 | Detection Level | Correction Level |
---|---|---
sighan13 | p:0.84,r:0.792,F1:0.815 | p:0.823,recall:0.775,F1:0.799 
new_data | p:0.491,r:0.398,F1:0.44 | p:0.442,recall:0.358,F1:0.396


- 注意
1. 先用wang2018数据预训练，再用同分布的sighan13微调
2. 使用词表的分词方式容易造成拼音和数字错误修正，因此直接使用一个字作为token，不使用基于词表的分词（csc_train_mlm_tok.py）



