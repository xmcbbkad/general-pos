# -*- coding: utf-8 -*-
# ontonotes raw data --> char+\t+tag(chinese)  word+\t+tag(english)

import re
import os
import sys
sys.path.append('../')
from bert_pos.bert import tokenization

basic_tokenizer = tokenization.BasicTokenizer(do_lower_case = False)

def should_split(last_char, char):
    if last_char == '' or char == '':
        return False

    if tokenization._is_punctuation(last_char) or tokenization._is_punctuation(char):
        return True

    if basic_tokenizer._is_chinese_char(ord(last_char)) or basic_tokenizer._is_chinese_char(ord(char)):
        return True

    return False

def change_format(fin_path, fout_path):
    fin = open(fin_path,'r')
    fout = open(fout_path,'a+')

    count = 0
    for line in fin:
        if line.strip() == '':
            fout.write('\n')
            continue

        pos_begin = line.rfind("(")
        pos_end = -1
        if pos_begin >=0:
            pos_end = line.find(")", pos_begin)

        if pos_begin >=0 and pos_end>0:
            str_pos = line[pos_begin+1: pos_end].strip()
            split_index = str_pos.find(' ')
            if str_pos[:split_index] != "-NONE-":
                word = str_pos[split_index+1:]
                label = str_pos[:split_index]
                
                this_count = 0
                tmp_chars = ''
                last_char = ''
                for i in range(len(word)):
                    if should_split(last_char, word[i]) == False:
                        tmp_chars += word[i]
                        last_char = word[i]
                        continue
                
                    if this_count == 0:
                        fout.write('%s\t%s\n' % (tmp_chars, 'B-'+label))
                    else:
                        fout.write('%s\t%s\n' % (tmp_chars, 'I-'+label))
                    
                    this_count += 1
                    tmp_chars = word[i]
                    last_char = word[i]

                if tmp_chars != '':
                    if this_count == 0:
                        fout.write('%s\t%s\n' % (tmp_chars, 'B-'+label))
                    else:
                        fout.write('%s\t%s\n' % (tmp_chars, 'I-'+label))


        else:
            print(line)

        
        count += 1
        if count % 100 == 0:
            print('=== %d sentence changed ===' % count)
    fin.close()
    fout.close()
    
    return True

####################################################################################
# traverse_datafile()函数用于遍历rootpath文件夹内的文件，
# 并找出其中后缀名为suffix的文件，对其进行格式转换并写入到fout_path路径指向的文件：
####################################################################################
def traverse_datafile(rootpath, suffix, fout_path):
    
    FILENAME_PATTERN = re.compile('^(.*?)' + suffix + '$')
    
    files = os.listdir(rootpath)
    for file in files:
        # print(file)
        if os.path.isdir(rootpath + '/' + file):
            # print(rootpath + '/' + file)
            traverse_datafile(rootpath + '/' + file, suffix, fout_path)
        else:
            if FILENAME_PATTERN.match(file):
                print(rootpath + '/' + file)
                change_format(rootpath + '/' + file, fout_path+file)
    
    return True

'''
==================================
=== 这一部分为实际的运行函数模块 ===
==================================
'''
if __name__ == '__main__':
    # === rootpath为ontonotes-release-5.0数据文件中所有的中文标注文件的存储目录根路径 ===
    rootpath = '/data/share/ontonotes-release-5.0/data/files/data/chinese/annotations'
    #rootpath = '/data/share/ontonotes-release-5.0/data/files/data/english/annotations'
    
    fout_path = '/data/jh/notebooks/fanxiaokun/code/general_pos/data/ontonotes_data/chinese_word_20190717/'

    suffix = '.parse'
    traverse_datafile(rootpath, suffix, fout_path)
