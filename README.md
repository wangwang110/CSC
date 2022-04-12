# ChineseBert_CSC
ChineseBert用于中文拼写纠错

数据来源:

训练数据 | 句子数 | 来源
---|---|---
wang2018 | 271329 | https://github.com/wdimmy/Automatic-Corpus-Generation
sighan13 | train:350 test:1000| http://nlp.ee.ncu.edu.tw/resource/csc.html
sighan14 | train:6526 test:1062 | http://nlp.ee.ncu.edu.tw/resource/csc.html
sighan15 | train:3174 test:1100 | http://nlp.ee.ncu.edu.tw/resource/csc.html



链接: https://pan.baidu.com/s/1Lr_L-lbesYW4fjhELtZ3-w?pwd=4863 提取码: 4863 



## 香侬科技ChineseBert用于中文拼写纠错

- 训练：

1. 拷贝ChineseBert代码https://github.com/ShannonAI/ChineseBert，
置于ChineseBert文件夹，并安装ChineseBert所需依赖

2. 下载训练数据，置于data文件夹，运行train.sh

注意：先用wang2018数据预训练，再用同分布的sighan13微调


- 测试：

运行eval.sh


- 纠正文本：

运行csc_eval.py 即可



已经训练好的模型：

sighan13链接：

sighan15链接：


- 指标：

数据集 | correct F1
---|---
sighan13 | 82.3
sighan15 | 75




