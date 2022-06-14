# -*- coding: utf-8 -*-
# ontonotes raw data --> char+\t+tag(chinese)  word+\t+tag(english)

import re
import os
import sys
sys.path.append('../')
from bert_pos.bert import tokenization

DO_LOWER_CASE = True
#DO_LOWER_CASE = False

vocab_file = "/data/jh/notebooks/fanxiaokun/code/general_pos/bert_pos/chinese_L-12_H-768_A-12/vocab.txt"
#vocab_file = "/data/jh/notebooks/fanxiaokun/code/general_pos/bert_pos/multi_cased_L-12_H-768_A-12/vocab.txt"
wordpiece_tokenizer = tokenization.WordpieceTokenizer(vocab=tokenization.load_vocab(vocab_file))

def change_format(fin_path, fout_path):
    fin = open(fin_path,'r')
    fout = open(fout_path,'a+')

    count = 0
    for line in fin:
        if line.strip() == '':
            fout.write('\n')
            continue
            
        s = line.strip().split('\t')
        word = s[0]
        tag = s[1]
        
        if DO_LOWER_CASE:
            word = word.lower()

        wordpiece = wordpiece_tokenizer.tokenize(word)
        if len(wordpiece) == 0:
            print("len(wordpiece)==0 word:{}".formate(word))
        fout.write("{}\t{}\n".format(wordpiece[0], tag))        
        for sub_word in wordpiece[1:]:
            fout.write("{}\t{}\n".format(sub_word, tag.replace('B-', 'I-'))) 

    fin.close()
    fout.close()
    
    return True

def traverse_datafile(rootpath, fout_path):
    files = os.listdir(rootpath)
    for file in files:
        # print(file)
        if os.path.isdir(rootpath + '/' + file):
            # print(rootpath + '/' + file)
            traverse_datafile(rootpath + '/' + file, fout_path)
        else:
            print(rootpath + '/' + file)
            change_format(rootpath + '/' + file, fout_path+file)
    
    return True

if __name__ == '__main__':
    
    rootpath = '/data/jh/notebooks/fanxiaokun/code/general_pos/data/ontonotes_data/chinese_word_merge_20190717/'
    
    fout_path = '/data/jh/notebooks/fanxiaokun/code/general_pos/data/ontonotes_data/chinese_word_piece_20190717/'

    traverse_datafile(rootpath, fout_path)
