# ChineseBert_CSC
ChineseBert用于中文拼写纠错


数据来源:

SIGHAN Bake-off 2013: SIGHAN Bake-off 2013: Chinese Spelling Check Task
SIGHAN Bake-off 2014: CLP-2014 Bake-off: Chinese Spelling Check Task
SIGHAN Bake-off 2015: SIGHAN Bake-off 2015: Chinese Spelling Check Task
Wang271K: https://github.com/wdimmy/Automatic-Corpus-Generation

sighan13，sighan14，sighan15 包含对应的训练集和测试集，Wang271K仅仅用来训练

具体可参考：https://github.com/DaDaMrX/ReaLiSe

处理过的数据链接: https://pan.baidu.com/s/1Lr_L-lbesYW4fjhELtZ3-w?pwd=4863 提取码: 4863 

## 榜单

各论文指标（不同的论文bert取得的结果还不一样）


数据集 | 模型 | Detection Level | Correction Level
---|---|---|---
SIGHAN13 | Hybrid (Wang et al., 2018) | 54.0 69.3 60.7 | - - 52.1
SIGHAN13 | FASpell (Hong et al., 2019)|76.2 63.2 69.1| 73.1 60.5 66.2
SIGHAN13 | SpellGCN (Cheng et al., 2020) | 80.1 74.4 77.2 | 78.3 72.7 75.4

SIGHAN13 | BERT | 85.0 77.0 80.8 | 83.0 75.2 78.9
SIGHAN13 | ECOPO (BERT)|87.2 81.7 84.4| 86.1 80.6 83.3
SIGHAN13 | REALISE | 88.6 82.5 85.4 | 87.2 81.2 84.1
SIGHAN13 | ECOPO (REALISE) |89.3 83.2 86.2| 88.5 82.0 85.1





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




