# ChineseBert_CSC
ChineseBert用于中文拼写纠错


数据来源:

SIGHAN Bake-off 2013: http://ir.itc.ntnu.edu.tw/lre/sighan7csc.html

SIGHAN Bake-off 2014: http://ir.itc.ntnu.edu.tw/lre/clp14csc.html

SIGHAN Bake-off 2015: http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html

Wang271K: https://github.com/wdimmy/Automatic-Corpus-Generation

sighan13，sighan14，sighan15 包含对应的训练集和测试集，Wang271K仅仅用来训练


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
 Google BERT | 79.0 72.8 75.8 | 77.7 71.6 74.6 |  [论文](https://arxiv.org/pdf/2004.14166.pdf)
 SpellGCN | 80.1 74.4 77.2 | 78.3 72.7 75.4 | [论文](https://arxiv.org/pdf/2004.14166.pdf)
 ERNIE  |76.6 71.9 74.2 | 73.0 68.5 70.6 | [论文](https://aclanthology.org/2021.findings-acl.198.pdf)
 MLM-phonetics | 82.0 78.3 80.1| 79.5 77.0 78.2 | [论文](https://aclanthology.org/2021.findings-acl.198.pdf)
 Google BERT  |98.7 70.6 82.3|98.6 67.8 80.4| [论文](https://aclanthology.org/2021.acl-long.464.pdf)
 PHMOSpell |99.5 76.8 86.7| 99.5 75.1 85.6 | [论文](https://aclanthology.org/2021.acl-long.464.pdf)
 BERT-wwm  | 85.0 77.0 80.8| 83.0 75.2 78.9 | [论文](https://aclanthology.org/2021.findings-acl.64.pdf)
 REALISE (BERT-wwm ) | 88.6 82.5 85.4 | 87.2 81.2 84.1 | [论文](https://aclanthology.org/2021.findings-acl.64.pdf)
 ECOPO BERT-wwm  |87.2 81.7 84.4 | 86.1 80.6 83.3|  [论文](https://arxiv.org/pdf/2203.00991.pdf) 
 ECOPO REALISE (BERT-wwm )|89.3 83.2 86.2 | 88.5 82.0 85.1| [论文](https://arxiv.org/pdf/2203.00991.pdf)

sighan14结果:

 模型 | Detection Level | Correction Level |来源
---|---|--- |---
 Hybrid | 51.9 66.2 58.2 | _ _ 56.1 | [论文](https://aclanthology.org/D18-1273.pdf)
 FASpell |61.0 53.5 57.0| 59.4 52.0 55.4 | [论文](https://aclanthology.org/D19-5522.pdf)
 Google BERT | 65.6 68.1 66.8 | 63.1 65.5 64.3 | [论文](https://arxiv.org/pdf/2004.14166.pdf)
 SpellGCN | 65.1 69.5 67.2 | 63.1 67.2 65.3 | [论文](https://arxiv.org/pdf/2004.14166.pdf)
 ERNIE  |63.5 69.3 66.3 | 60.1 65.6 62.8| [论文](https://aclanthology.org/2021.findings-acl.198.pdf)
 MLM-phonetics | 66.2 73.8 69.8| 64.2 73.8 68.7 | [论文](https://aclanthology.org/2021.findings-acl.198.pdf)
 Google BERT  |78.6 60.7 68.5|77.8 57.6 66.2| [论文](https://aclanthology.org/2021.acl-long.464.pdf)
 PHMOSpell |85.3 67.6 75.5| 84.7 64.3 73.1 |[论文](https://aclanthology.org/2021.acl-long.464.pdf)
 Google BERT(4 layer) |82.6 59.0 68.8| 82.4 58.0 68.1| [论文](https://aclanthology.org/2021.emnlp-main.287v2.pdf)
 SpellBERT (4 layer) |83.1 62.0 71.0| 82.9 61.2 70.4| [论文](https://aclanthology.org/2021.emnlp-main.287v2.pdf)
 BERT-wwm  | 64.5 68.6 66.5| 62.4 66.3 64.3| [论文](https://aclanthology.org/2021.findings-acl.64.pdf)
 REALISE (BERT-wwm )| 67.8 71.5 69.6 | 66.3 70.0 68.1 | [论文](https://aclanthology.org/2021.findings-acl.64.pdf)
 ECOPO BERT-wwm  |65.8 69.0 67.4 | 63.7 66.9 65.3|  [论文](https://arxiv.org/pdf/2203.00991.pdf) 
 ECOPO REALISE (BERT-wwm ) |68.8 72.1 70.4 | 67.5 71.0 69.2| [论文](https://arxiv.org/pdf/2203.00991.pdf)



sighan15结果:

 模型 | Detection Level | Correction Level | 来源
---|---|---|---
 Hybrid | 56.6 69.4 62.3 |  _ _ 57.1  | [论文](https://aclanthology.org/D18-1273.pdf)
 FASpell | 67.6 60.0 63.5 | 66.6 59.1 62.6 | [论文](https://aclanthology.org/D19-5522.pdf)
 Google BERT | 73.0 70.8 71.9 | 65.9 64.0 64.9 | [论文](https://arxiv.org/pdf/2005.07421.pdf)
 Soft-Masked BERT | 73.7 73.2 73.5 | 66.7 66.2 66.4 | [论文](https://arxiv.org/pdf/2005.07421.pdf)
 Google BERT | 73.7 78.2 75.9 | 70.9 75.2 73.0 | [论文](https://arxiv.org/pdf/2004.14166.pdf)
 SpellGCN | 74.8 80.7 77.7 | 72.1 77.7 75.9 | [论文](https://arxiv.org/pdf/2004.14166.pdf)
 ERNIE  |73.6 79.8 76.6 | 68.6 74.4 71.4| [论文](https://aclanthology.org/2021.findings-acl.198.pdf)
 MLM-phonetics | 77.5 83.1 80.2| 74.9 80.2 77.5 |[论文](https://aclanthology.org/2021.findings-acl.198.pdf)
 Google BERT  |68.4 77.6 72.7 | 66.0 74.9 70.2| [论文](https://aclanthology.org/2021.acl-long.233.pdf)
 PLOME | 77.4 81.5 79.4 | 75.3 79.3 77.2 | [论文](https://aclanthology.org/2021.acl-long.233.pdf)
 Google BERT  |79.9 84.1 72.9 78.1|83.1 68.0 74.8| [论文](https://aclanthology.org/2021.acl-long.464.pdf)
 PHMOSpell |90.1 72.7 80.5| 89.6 69.2 78.1| [论文](https://aclanthology.org/2021.acl-long.464.pdf)
 Google BERT(4 layer) |85.2 68.9 76.2| 84.8 66.9 74.8| [论文](https://aclanthology.org/2021.emnlp-main.287v2.pdf)
 SpellBERT (4 layer) |87.5 73.6 80.0| 87.1 71.5 78.5| [论文](https://aclanthology.org/2021.emnlp-main.287v2.pdf)
 BERT-wwm  | 74.2 78.0 76.1 | 71.6 75.3 73.4 | [论文](https://aclanthology.org/2021.findings-acl.64.pdf)
 REALISE (BERT-wwm)  | 77.3 81.3 79.3 | 75.9 79.9 77.8 | [论文](https://aclanthology.org/2021.findings-acl.64.pdf)
 ECOPO BERT-wwm |78.2 82.3 80.2| 76.6 80.4 78.4 |  [论文](https://arxiv.org/pdf/2203.00991.pdf) 
 ECOPO REALISE (BERT-wwm) |77.5 82.6 80.0| 76.1 81.2 78.5| [论文](https://arxiv.org/pdf/2203.00991.pdf)
 
 
 此外，还有一篇通过数据增强的方法来处理csc任务的论文
 
 [Exploration and Exploitation: Two Ways to Improve Chinese Spelling Correction Model](https://arxiv.org/pdf/2105.14813.pdf)
 
 只不过该论文是在不同的训练集上训练，得到不同数据集对应的模型。例如，在sighan15训练的模型，用于sighan15测试集的测试，所以结果会相对偏高
 
## 香侬科技ChineseBert用于中文拼写纠错

ChineseBert repo[https://github.com/ShannonAI/ChineseBert]

![image](https://user-images.githubusercontent.com/21475557/164133645-e560f580-4c54-4f38-9bca-153f6bf0fd32.png)

- 训练：

1. 下载ChineseBert放出的预训练模型，放置在本地文件夹（chinese_bert_path 参数）

2. 拷贝ChineseBert代码，置于ChineseBert文件夹，并安装ChineseBert所需依赖

3. 下载训练数据，置于data文件夹，运行train.sh


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



