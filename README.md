# ChineseBert_CSC
ChineseBert用于中文拼写纠错


数据来源:

SIGHAN Bake-off 2013: http://ir.itc.ntnu.edu.tw/lre/sighan7csc.html

SIGHAN Bake-off 2014: http://ir.itc.ntnu.edu.tw/lre/clp14csc.html

SIGHAN Bake-off 2015: http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html

Wang271K: https://github.com/wdimmy/Automatic-Corpus-Generation

sighan13，sighan14，sighan15 包含对应的训练集和测试集，Wang271K仅仅用来训练

具体可参考：https://github.com/DaDaMrX/ReaLiSe

处理过的数据链接: https://pan.baidu.com/s/1Lr_L-lbesYW4fjhELtZ3-w?pwd=4863 提取码: 4863 


## 评价指标
相比官方放出的评价指标，该指标更为严格（所有修正过的句子都算作P值的分母）

## 榜单

各论文指标（不同的论文使用bert取得的结果不一样）

sighan13结果:

 模型 | Detection Level | Correction Level | 来源
---|---|--- |---
 Hybrid | 54.0 69.3 60.7 |  _ _ 52.1 | [论文](https://aclanthology.org/D18-1273.pdf)
 FASpell (Hong et al., 2019)|76.2 63.2 69.1| 73.1 60.5 66.2 | 论文
 BERT (Cheng et al., 2020) | 79.0 72.8 75.8 | 77.7 71.6 74.6 | 论文
 SpellGCN (Cheng et al., 2020) | 80.1 74.4 77.2 | 78.3 72.7 75.4 | 论文
 ERNIE  |76.6 71.9 74.2 | 73.0 68.5 70.6 | 论文[https://aclanthology.org/2021.findings-acl.198.pdf]
 MLM-phonetics | 82.0 78.3 80.1| 79.5 77.0 78.2 | 论文[https://aclanthology.org/2021.findings-acl.198.pdf]
 BERT (Xu et al., 2021) | 85.0 77.0 80.8 | 83.0 75.2 78.9 | 论文
 ECOPO BERT(Li et al.,2022)|87.2 81.7 84.4| 86.1 80.6 83.3 | 论文
 REALISE (Xu et al., 2021) | 88.6 82.5 85.4 | 87.2 81.2 84.1 | 论文
 BERT (Huang et al., 2021) | 98.7 70.6 82.3 | 67.8 98.6 67.8 80.4 | 论文
 PHMOSpell (Huang et al., 2021) | 99.5 76.8 86.7 | 99.5 75.1 85.6 | 论文
 ECOPO REALISE Li et al.,2022) |89.3 83.2 86.2| 88.5 82.0 85.1 | 论文


sighan14结果:

 模型 | Detection Level | Correction Level |来源
---|---|--- |---
 Hybrid | 51.9 66.2 58.2 | _ _ 56.1 | [论文](https://aclanthology.org/D18-1273.pdf)
 FASpell (Hong et al., 2019)|61.0 53.5 57.0| 59.4 52.0 55.4 | 论文
 BERT (Cheng et al., 2020) | 65.6 68.1 66.8 | 63.1 65.5 64.3 | 论文
 SpellGCN (Cheng et al., 2020) | 65.1 69.5 67.2 | 63.1 67.2 65.3 | 论文
 
 ERNIE  |63.5 69.3 66.3 | 60.1 65.6 62.8| 论文[https://aclanthology.org/2021.findings-acl.198.pdf]
 MLM-phonetics | 66.2 73.8 69.8| 64.2 73.8 68.7 | 论文[https://aclanthology.org/2021.findings-acl.198.pdf]
 
 BERT (Xu et al., 2021) | 85.0 77.0 80.8 | 83.0 75.2 78.9 | 论文
 ECOPO BERT(Li et al.,2022)|87.2 81.7 84.4| 86.1 80.6 83.3 | 论文
 REALISE (Xu et al., 2021) | 88.6 82.5 85.4 | 87.2 81.2 84.1 | 论文
 BERT (Huang et al., 2021) | 98.7 70.6 82.3 | 67.8 98.6 67.8 80.4 | 论文
 PHMOSpell (Huang et al., 2021) | 99.5 76.8 86.7 | 99.5 75.1 85.6 | 论文
 ECOPO REALISE Li et al.,2022) |89.3 83.2 86.2| 88.5 82.0 85.1 | 论文


sighan15结果:

 模型 | Detection Level | Correction Level | 来源
---|---|---|---
 Hybrid (Wang et al., 2018) | 56.6 69.4 62.3 |  _ _ 57.1  | [论文](https://aclanthology.org/D18-1273.pdf)
 FASpell (Hong et al., 2019) | 67.6 60.0 63.5 | 66.6 59.1 62.6 | 论文
 BERT (Zhang, et al., 2020) | 73.0 70.8 71.9 | 65.9 64.0 64.9 | 论文 
 Soft-Masked BERT Zhang, et al., 2020) | 73.7 73.2 73.5 | 66.7 66.2 66.4 | 论文
 BERT (Cheng et al., 2020) | 73.7 78.2 75.9 | 70.9 75.2 73.0 | 论文
 SpellGCN (Cheng et al., 2020) | 74.8 80.7 77.7 | 72.1 77.7 75.9 | 论文
 
 ERNIE  |73.6 79.8 76.6 | 68.6 74.4 71.4| 论文[https://aclanthology.org/2021.findings-acl.198.pdf]
 MLM-phonetics | 77.5 83.1 80.2| 74.9 80.2 77.5 | 论文[https://aclanthology.org/2021.findings-acl.198.pdf]
 
 BERT (Xu et al., 2021) | 85.0 77.0 80.8 | 83.0 75.2 78.9 | 论文
 ECOPO BERT(Li et al.,2022)|87.2 81.7 84.4| 86.1 80.6 83.3 | 论文 
 REALISE (Xu et al., 2021) | 88.6 82.5 85.4 | 87.2 81.2 84.1 | 论文
 BERT (Huang et al., 2021) | 98.7 70.6 82.3 | 67.8 98.6 67.8 80.4 | 论文
 PHMOSpell (Huang et al., 2021) | 99.5 76.8 86.7 | 99.5 75.1 85.6 | 论文
 ECOPO REALISE Li et al.,2022) |89.3 83.2 86.2| 88.5 82.0 85.1 | 论文
 
 
 
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




