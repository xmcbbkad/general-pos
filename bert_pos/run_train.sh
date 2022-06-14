CUDA_VISIBLE_DEVICES=0 python -u train.py \
--log_root='log_chinese_20190717_maxlength_192' \
--bert_config_file='chinese_L-12_H-768_A-12/bert_config.json' \
--init_checkpoint='chinese_L-12_H-768_A-12/bert_model.ckpt' \
--vocab_path='chinese_L-12_H-768_A-12/vocab.txt' \
--train_file_path='general_pos/data/ontonotes_data/chinese_word_piece_20190717/data_train' \
--dev_file_path='general_pos/data/ontonotes_data/chinese_word_piece_20190717/data_test' \
--max_seq_length=192 \
