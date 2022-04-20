

train_data=./data/13train.txt
save_path=./save/wang2018/
chinese_bert_path=/home/plm_models/ChineseBERT-base/

CUDA_VISIBLE_DEVICES="1" python  csc_train_mlm_tok.py --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 \
--load_model=False --do_train=True --chinese_bert_path=$chinese_bert_path  --train_data=$train_data  --epoch=10 --batch_size=10 --learning_rate=2e-5 \
 --do_save=True --save_dir=$save_path --seed=10 > $save_path/csc_train_mlm.log 2>&1 &

# csc_train_mlm_tok.py 分词方式改变了，直接将一个字符看作是一个词，不使用词表分词
# csc_train_mlm.py 使用词表分词,该方法会造成拼音和数字的错误纠正