



test_data=./data/13test.txt
model_path=./save/wang2018/sighan13/model.pkl
chinese_bert_path=/home/plm_models/ChineseBERT-base/

CUDA_VISIBLE_DEVICES="1" python  csc_train_mlm_tok.py --task_name=test  --gpu_num=1 --do_train=False  --load_model=True \
--load_path=$model_path --chinese_bert_path=$chinese_bert_path --do_test=True --test_data=$test_data --batch_size=16
