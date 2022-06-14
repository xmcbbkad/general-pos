# -*- coding: utf-8 -*-
import os,sys
#import tokenization
from general_pos_predict_sdk import tokenization    

def is_blank_char(char):
    if tokenization._is_whitespace(char):
        return True
    
    cp = ord(char)
    if cp == 0 or cp == 0xfffd:
        return True
 
    if tokenization._is_control(char):
        return True
    
    return False

def should_split(chars, char, basic_tokenizer):
    if chars == '' or char == '':
        return False

    last_is_split = is_blank_char(chars[-1])
    this_is_split = is_blank_char(char)
    if last_is_split != this_is_split:
        return True

    if tokenization._is_punctuation(chars[-1]):
        return True
    if tokenization._is_punctuation(char):
        return True

    last_is_chinese = basic_tokenizer._is_chinese_char(ord(chars[-1]))
    this_is_chinese = basic_tokenizer._is_chinese_char(ord(char))
    if last_is_chinese and this_is_chinese:
        return True
    if not last_is_chinese and this_is_chinese:
        return True
    if last_is_chinese and not this_is_chinese:
        return True

    last_is_digit = chars[-1].isdigit()
    this_is_digit = char.isdigit()
    if last_is_digit and not this_is_digit:
        return True
    if not last_is_digit and this_is_digit:
        return True

    return False

class GroupText():
    def __init__(self, text):
        self.text = text
        self.is_blank = False
        self.tag = ''
        self.index = -1
        self.offset = -1
        self.length = len(text)
        self.word_piece = []
        self.sum_before_word_piece = 0

class SentenceTokenizer():
    def __init__(self, text, wordpiece_tokenizer, basic_tokenizer, do_lower_case=True):
        self.do_lower_case = do_lower_case
        self.text_raw = text
        self.wordpiece_tokenizer = wordpiece_tokenizer
        self.basic_tokenizer = basic_tokenizer
        self.group = self.tokenize(text)
        self.list_wordpiece = []
        self.list_wordpiece_index = [] 
        self.list_wordpiece_group_index = []
        self.build_index(self.group)
    
    def build_index(self,group):
        for i in range(len(group)):
            if group[i].is_blank:
                continue
            for j in range(len(group[i].word_piece)):
                self.list_wordpiece.append(group[i].word_piece[j])
                self.list_wordpiece_index.append(group[i].sum_before_word_piece + j)
                self.list_wordpiece_group_index.append(group[i].index)

    def tokenize(self, text):
        list_group = []

        #if self.do_lower_case:
        #    text = text.lower()
        chars = list(text)

        tmp_chars = ''
        group_index = 0
        group_offset = 0
        group_sum_before_word_piece = 0
        
        for i in range(len(chars)):
            if should_split(tmp_chars, chars[i], self.basic_tokenizer):
                group_item = GroupText(tmp_chars)
                if is_blank_char(tmp_chars[-1]):
                    group_item.is_blank = True
                group_item.index = group_index
                group_item.offset = group_offset
                group_item.word_piece = self.wordpiece_tokenizer.tokenize(tmp_chars.lower() if self.do_lower_case else tmp_chars)    
                group_item.sum_before_word_piece = group_sum_before_word_piece
 
                list_group.append(group_item)
                
                tmp_chars = chars[i]
                group_index += 1
                group_offset = i
                group_sum_before_word_piece += len(group_item.word_piece)

            else:
                tmp_chars += chars[i]

        if tmp_chars != '':
            group_item = GroupText(tmp_chars)
            if is_blank_char(tmp_chars[-1]):
                group_item.is_blank = True
            group_item.index = group_index
            group_item.offset = group_offset
            group_item.word_piece = self.wordpiece_tokenizer.tokenize(tmp_chars.lower() if self.do_lower_case else tmp_chars)    
            group_item.sum_before_word_piece = group_sum_before_word_piece
            list_group.append(group_item)
        
        return list_group

if __name__ == '__main__':
    vocab_file = "data/chinese_vocab.txt"
    wordpiece_tokenizer = tokenization.WordpieceTokenizer(vocab = tokenization.load_vocab(vocab_file))
    basic_tokenizer = tokenization.BasicTokenizer(do_lower_case=True)
    full_tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    text = "我们都是中国 人，  hi how old aregefgge ffff"
    text = "\n\t   \t\n... . . . 我们都是中国 人，  hi how old. i\,bf  aregefgge ffff, i have b23705s, aa,她的名字叫sasa liu。"

    text = "ACTOYS2014年才开始正式商业化运作"
    sentence_tokens = SentenceTokenizer(text=text, wordpiece_tokenizer=wordpiece_tokenizer, basic_tokenizer=basic_tokenizer, do_lower_case=True)
    
    print(text)

    #print("basic_tokenize:{}".format(basic_tokenizer.tokenize(text)))
    #print("full_tokenize:{}".format(full_tokenizer.tokenize(text)))

    #for item in sentence_tokens.group:
    #    print("is_blank:{},len:{},tokens:{}".format(item.is_blank, item.length, item.text))


    for item in sentence_tokens.group:
        print("text:{}--is_blank:{}--index:{}--offset:{}--length:{}--sum_before_word_piece:{}--word_piece:{}".format(item.text, item.is_blank, item.index, item.offset, item.length, item.sum_before_word_piece, item.word_piece)) 
    
    for i in range(len(sentence_tokens.list_wordpiece)):
        print("wordpiece:{}--index:{}--group_index:{}".format(sentence_tokens.list_wordpiece[i], sentence_tokens.list_wordpiece_index[i], sentence_tokens.list_wordpiece_group_index[i]))
