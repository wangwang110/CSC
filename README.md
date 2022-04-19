# ChineseBert_CSC
ChineseBert用于中文拼写纠错


数据来源:

SIGHAN Bake-off 2013: http://ir.itc.ntnu.edu.tw/lre/sighan7csc.html

SIGHAN Bake-off 2014: http://ir.itc.ntnu.edu.tw/lre/clp14csc.html

SIGHAN Bake-off 2015: http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html

Wang271K: https://github.com/wdimmy/Automatic-Corpus-Generation

sighan13，sighan14，sighan15 包含对应的训练集和测试集，Wang271K仅仅用来训练

如何处理，可参考：https://github.com/DaDaMrX/ReaLiSe

处理过的数据链接: https://pan.baidu.com/s/1Lr_L-lbesYW4fjhELtZ3-w?pwd=4863 


## 评价指标
[代码](https://github.com/wangwang110/ChineseBert_CSC/blob/main/ChineseBert/getF1.py)

相比[官方](http://nlp.ee.ncu.edu.tw/resource/csc.html)放出的评价指标，该指标更为严格（所有修正过的句子都算作P值的分母），论文大都使用该评价指标


## 榜单

各论文指标（不同的论文使用bert取得的结果不一样）

sighan13结果:

 模型 | Detection Level | Correction Level | 来源
---|---|--- |---
 Hybrid | 54.0 69.3 60.7 |  _ _ 52.1 | [论文](https://aclanthology.org/D18-1273.pdf)
 FASpell |76.2 63.2 69.1| 73.1 60.5 66.2 | [论文](https://aclanthology.org/D19-5522.pdf)
 BERT | 79.0 72.8 75.8 | 77.7 71.6 74.6 |  [论文](https://arxiv.org/pdf/2004.14166.pdf)
 SpellGCN | 80.1 74.4 77.2 | 78.3 72.7 75.4 | [论文](https://arxiv.org/pdf/2004.14166.pdf)
 ERNIE  |76.6 71.9 74.2 | 73.0 68.5 70.6 | [论文](https://aclanthology.org/2021.findings-acl.198.pdf)
 MLM-phonetics | 82.0 78.3 80.1| 79.5 77.0 78.2 | [论文](https://aclanthology.org/2021.findings-acl.198.pdf)
 BERT  |98.7 70.6 82.3|98.6 67.8 80.4| [论文](https://aclanthology.org/2021.acl-long.464.pdf)
 PHMOSpell |99.5 76.8 86.7| 99.5 75.1 85.6 | [论文](https://aclanthology.org/2021.acl-long.464.pdf)
 BERT | 85.0 77.0 80.8| 83.0 75.2 78.9 | [论文](https://aclanthology.org/2021.findings-acl.64.pdf)
 REALISE | 88.6 82.5 85.4 | 87.2 81.2 84.1 | [论文](https://aclanthology.org/2021.findings-acl.64.pdf)
 ECOPO BERT |87.2 81.7 84.4 | 86.1 80.6 83.3|  [论文](https://arxiv.org/pdf/2203.00991.pdf) 
 ECOPO REALISE |89.3 83.2 86.2 | 88.5 82.0 85.1| [论文](https://arxiv.org/pdf/2203.00991.pdf)

sighan14结果:

 模型 | Detection Level | Correction Level |来源
---|---|--- |---
 Hybrid | 51.9 66.2 58.2 | _ _ 56.1 | [论文](https://aclanthology.org/D18-1273.pdf)
 FASpell |61.0 53.5 57.0| 59.4 52.0 55.4 | [论文](https://aclanthology.org/D19-5522.pdf)
 BERT | 65.6 68.1 66.8 | 63.1 65.5 64.3 | [论文](https://arxiv.org/pdf/2004.14166.pdf)
 SpellGCN | 65.1 69.5 67.2 | 63.1 67.2 65.3 | [论文](https://arxiv.org/pdf/2004.14166.pdf)
 ERNIE  |63.5 69.3 66.3 | 60.1 65.6 62.8| [论文](https://aclanthology.org/2021.findings-acl.198.pdf)
 MLM-phonetics | 66.2 73.8 69.8| 64.2 73.8 68.7 | [论文](https://aclanthology.org/2021.findings-acl.198.pdf)
 BERT  |78.6 60.7 68.5|77.8 57.6 66.2| [论文](https://aclanthology.org/2021.acl-long.464.pdf)
 PHMOSpell |85.3 67.6 75.5| 84.7 64.3 73.1 |[论文](https://aclanthology.org/2021.acl-long.464.pdf)
 BERT(4 layer) |82.6 59.0 68.8| 82.4 58.0 68.1| [论文](https://aclanthology.org/2021.emnlp-main.287v2.pdf)
 SpellBERT (4 layer) |83.1 62.0 71.0| 82.9 61.2 70.4| [论文](https://aclanthology.org/2021.emnlp-main.287v2.pdf)
 BERT | 64.5 68.6 66.5| 62.4 66.3 64.3| [论文](https://aclanthology.org/2021.findings-acl.64.pdf)
 REALISE | 67.8 71.5 69.6 | 66.3 70.0 68.1 | [论文](https://aclanthology.org/2021.findings-acl.64.pdf)
 ECOPO BERT |65.8 69.0 67.4 | 63.7 66.9 65.3|  [论文](https://arxiv.org/pdf/2203.00991.pdf) 
 ECOPO REALISE |68.8 72.1 70.4 | 67.5 71.0 69.2| [论文](https://arxiv.org/pdf/2203.00991.pdf)



sighan15结果:

 模型 | Detection Level | Correction Level | 来源
---|---|---|---
 Hybrid | 56.6 69.4 62.3 |  _ _ 57.1  | [论文](https://aclanthology.org/D18-1273.pdf)
 FASpell | 67.6 60.0 63.5 | 66.6 59.1 62.6 | [论文](https://aclanthology.org/D19-5522.pdf)
 BERT | 73.0 70.8 71.9 | 65.9 64.0 64.9 | [论文](https://arxiv.org/pdf/2005.07421.pdf)
 Soft-Masked BERT | 73.7 73.2 73.5 | 66.7 66.2 66.4 | [论文](https://arxiv.org/pdf/2005.07421.pdf)
 BERT | 73.7 78.2 75.9 | 70.9 75.2 73.0 | [论文](https://arxiv.org/pdf/2004.14166.pdf)
 SpellGCN | 74.8 80.7 77.7 | 72.1 77.7 75.9 | [论文](https://arxiv.org/pdf/2004.14166.pdf)
 ERNIE  |73.6 79.8 76.6 | 68.6 74.4 71.4| [论文](https://aclanthology.org/2021.findings-acl.198.pdf)
 MLM-phonetics | 77.5 83.1 80.2| 74.9 80.2 77.5 |[论文](https://aclanthology.org/2021.findings-acl.198.pdf)
 BERT  |68.4 77.6 72.7 | 66.0 74.9 70.2| [论文](https://aclanthology.org/2021.acl-long.233.pdf)
 PLOME | 77.4 81.5 79.4 | 75.3 79.3 77.2 | [论文](https://aclanthology.org/2021.acl-long.233.pdf)
 BERT  |79.9 84.1 72.9 78.1|83.1 68.0 74.8| [论文](https://aclanthology.org/2021.acl-long.464.pdf)
 PHMOSpell |90.1 72.7 80.5| 89.6 69.2 78.1| [论文](https://aclanthology.org/2021.acl-long.464.pdf)
 BERT(4 layer) |85.2 68.9 76.2| 84.8 66.9 74.8| [论文](https://aclanthology.org/2021.emnlp-main.287v2.pdf)
 SpellBERT (4 layer) |87.5 73.6 80.0| 87.1 71.5 78.5| [论文](https://aclanthology.org/2021.emnlp-main.287v2.pdf)
 BERT  | 74.2 78.0 76.1 | 71.6 75.3 73.4 | [论文](https://aclanthology.org/2021.findings-acl.64.pdf)
 REALISE  | 77.3 81.3 79.3 | 75.9 79.9 77.8 | [论文](https://aclanthology.org/2021.findings-acl.64.pdf)
 ECOPO BERT |78.2 82.3 80.2| 76.6 80.4 78.4 |  [论文](https://arxiv.org/pdf/2203.00991.pdf) 
 ECOPO REALISE |77.5 82.6 80.0| 76.1 81.2 78.5| [论文](https://arxiv.org/pdf/2203.00991.pdf)
 
 
 
## 香侬科技ChineseBert用于中文拼写纠错

- 训练：
ChineseBert repo[https://github.com/ShannonAI/ChineseBert]

1. 下载ChineseBert放出的预训练模型

2. 拷贝ChineseBert代码，置于ChineseBert文件夹，并安装ChineseBert所需依赖

3. 下载训练数据，置于data文件夹，运行train.sh

注意：先用wang2018数据预训练，再用同分布的sighan13微调


- 测试：

运行eval.sh


- 纠正文本：

运行csc_eval.py 即可



- 已经训练好的模型：

wang2018数据训练
链接：

sighan13微调
链接：

sighan15微调
链接：


- 指标：

数据集 | correct F1
---|---
sighan13 | 82.3
sighan15 | 75




