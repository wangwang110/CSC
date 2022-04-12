



test_data=./data/13test.txt
model_path=./save/wang2018/sighan13/model.pkl

CUDA_VISIBLE_DEVICES="1" python  csc_train_mlm.py --task_name=test  --gpu_num=1 --do_train=False  --load_model=True \
--load_path=$model_path --do_test=True --test_data=$test_data --batch_size=16

 # csc_train_mlm_tok.py 分词方式改变了，直接将一个字符看作是一个词，不使用词表分词
