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

 模型 | Detection Level | Correction Level
---|---|---
 Hybrid (Wang et al., 2018) | 54.0 69.3 60.7 |  _ _ 52.1
 FASpell (Hong et al., 2019)|76.2 63.2 69.1| 73.1 60.5 66.2
 SpellGCN (Cheng et al., 2020) | 80.1 74.4 77.2 | 78.3 72.7 75.4
 BERT (Xu et al., 2021) | 85.0 77.0 80.8 | 83.0 75.2 78.9
 ECOPO BERT(Li et al.,2022)|87.2 81.7 84.4| 86.1 80.6 83.3
 REALISE (Xu et al., 2021) | 88.6 82.5 85.4 | 87.2 81.2 84.1
 BERT (Huang et al., 2021) | 98.7 70.6 82.3 | 67.8 98.6 67.8 80.4
 PHMOSpell (Huang et al., 2021) | 99.5 76.8 86.7 | 99.5 75.1 85.6
 ECOPO REALISE Li et al.,2022) |89.3 83.2 86.2| 88.5 82.0 85.1


sighan14结果:

 模型 | Detection Level | Correction Level
---|---|---
 Hybrid (Wang et al., 2018) | 51.9 66.2 58.2 | _ _ 56.1
 FASpell (Hong et al., 2019)|61.0 53.5 57.0| 59.4 52.0 55.4
 SpellGCN (Cheng et al., 2020) | 80.1 74.4 77.2 | 78.3 72.7 75.4
 BERT (Xu et al., 2021) | 85.0 77.0 80.8 | 83.0 75.2 78.9
 ECOPO BERT(Li et al.,2022)|87.2 81.7 84.4| 86.1 80.6 83.3
 REALISE (Xu et al., 2021) | 88.6 82.5 85.4 | 87.2 81.2 84.1
 BERT (Huang et al., 2021) | 98.7 70.6 82.3 | 67.8 98.6 67.8 80.4
 PHMOSpell (Huang et al., 2021) | 99.5 76.8 86.7 | 99.5 75.1 85.6
 ECOPO REALISE Li et al.,2022) |89.3 83.2 86.2| 88.5 82.0 85.1


sighan15结果:

 模型 | Detection Level | Correction Level
---|---|---
 Hybrid (Wang et al., 2018) | 56.6 69.4 62.3 |  _ _ 57.1
 FASpell (Hong et al., 2019) | 67.6 60.0 63.5 | 66.6 59.1 62.6
 SpellGCN (Cheng et al., 2020) | 80.1 74.4 77.2 | 66.6 59.1 62.6
 BERT (Xu et al., 2021) | 85.0 77.0 80.8 | 83.0 75.2 78.9
 ECOPO BERT(Li et al.,2022)|87.2 81.7 84.4| 86.1 80.6 83.3
 REALISE (Xu et al., 2021) | 88.6 82.5 85.4 | 87.2 81.2 84.1
 BERT (Huang et al., 2021) | 98.7 70.6 82.3 | 67.8 98.6 67.8 80.4
 PHMOSpell (Huang et al., 2021) | 99.5 76.8 86.7 | 99.5 75.1 85.6
 ECOPO REALISE Li et al.,2022) |89.3 83.2 86.2| 88.5 82.0 85.1
 
 
 
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




