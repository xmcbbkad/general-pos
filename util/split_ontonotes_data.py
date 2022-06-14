import re
import os
from random import random, gauss

def write_file(output_file, input_file):
    f_input = open(input_file, 'r')
    f_output = open(output_file, 'a+')

    for line in f_input:
        f_output.write(line)

    f_input.close()
    f_output.close()

######################################################################################################
# split_dataset()函数用于对数据文件进行切分
# fin_path为总的数据集的路径地址
# output_file_path为生成的训练集(train)文件，验证集(dev)文件以及测试集(test)文件所要保存的文件夹路径
# threshold为转换阈值，当前认为训练集，测试集以及验证集中的句子数概率分布为：
# (1-3*threshold) : (2*threshold) : threshold
# 默认threshold = 0.1，即三者比例为：7：2：1
######################################################################################################
def split_dataset(input_dir, output_dir, threshold = 0.1):
    files = os.listdir(input_dir)
    
    print("all file num={}".format(len(files)))
    count_all = 0
    count_train = 0
    count_dev = 0
    count_test = 0

    for file in files:
        seed = random()
        if seed < 1 - 2 * threshold:
            write_file(output_dir+'data_train', input_dir+file)
            count_train += 1
        elif seed < 1 - threshold:
            write_file(output_dir+'data_test', input_dir+file)
            count_test += 1
        else:
            write_file(output_dir+'data_dev', input_dir+file)
            count_dev += 1
        count_all += 1
        if count_all % 10 == 0:
            print('=== load %d files ===' % count_all)
    
    print('=== Totally load %d files ===' % count_all)
    print('\t=== The Distribution is: ===')
    print('\t=== Train : %d files ===' % count_train)
    print('\t=== Test  : %d files ===' % count_test)
    print('\t=== Dev   : %d files ===' % count_dev)
    
    return True

if __name__ == '__main__':
   
    input_dir = '/data/jh/notebooks/fanxiaokun/code/general_pos/data/ontonotes_data/chinese_word_20190717/'
    output_dir = '/data/jh/notebooks/fanxiaokun/code/general_pos/data/ontonotes_data/chinese_word_merge_20190717/'


    threshold = 0.1
    split_dataset(input_dir, output_dir, threshold)
