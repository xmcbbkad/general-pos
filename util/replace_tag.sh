#!/bin/bash

replace_dir="/data/jh/notebooks/fanxiaokun/code/general_pos/data/ontonotes_data/chinese_word_20190717/"

sed -i "s/NT-SHORT/NT/g" `grep NT-SHORT -rl $replace_dir` 
sed -i "s/NR-SHORT/NR/g" `grep NR-SHORT -rl $replace_dir`
sed -i "s/NN-SHORT/NN/g" `grep NN-SHORT -rl $replace_dir`
sed -i "s/INF/PN/g" `grep INF -rl $replace_dir` 

