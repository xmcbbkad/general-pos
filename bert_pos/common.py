# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import os

LABEL_LIST = ['O', 'B-NN', 'I-NN', 'B-PU', 'I-PU', 'B-VV', 'I-VV', 'B-AD', 'I-AD', 'B-NR', 'I-NR', 'B-PN', 'I-PN', 'B-P', 'I-P', 'B-CD', "I-CD", "B-DEG", "I-DEG", "B-M", "I-M", "B-JJ", "I-JJ", "B-DEC", "I-DEC", "B-DT", "I-DT", "B-VC", "I-VC", "B-VA", "I-VA", "B-NT", "I-NT", "B-LC", "I-LC", "B-SP", "I-SP", "B-AS", "I-AS", "B-CC", "I-CC", "B-VE", "I-VE", "B-IJ", "I-IJ", "B-OD", "I-OD", "B-CS", "I-CS", "B-MSP", "I-MSP", "B-DEV", "I-DEV", "B-BA", "I-BA", "B-ETC", "I-ETC", "B-SB", "I-SB", "B-DER", "I-DER", "B-LB", "I-LB", "B-URL", "I-URL", "B-FW", "I-FW", "B-ON", "I-ON", "B-X", "I-X"]

LABEL_MAP = {}
for (i, label) in enumerate(LABEL_LIST):
    LABEL_MAP[label] = i


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def convert_single_example(example, max_seq_length, tokenizer):
    tokens = ['[CLS]']
    labels = [LABEL_MAP['O']]

    for word,tag in zip(example.text_a, example.label):
        tokens.append(word)
        labels.append(LABEL_MAP[tag])


    if len(labels) != len(tokens):
        print("convert_single_example fail")
        return [], [], [], []


    if len(tokens) > max_seq_length-2:
        tokens = tokens[0:(max_seq_length-2)]
        labels = labels[0:(max_seq_length-2)]

    tokens.append('[SEP]')
    labels.append(LABEL_MAP['O'])
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        labels.append(LABEL_MAP['O'])

    return input_ids, input_mask, segment_ids, labels



def generate_batch(data, batch_size, max_seq_length, tokenizer, infinity = True):
    if infinity:
        while True:
            #imb_orgin = np.random.randint(0, len(data), batch_size*3)
            imb_orgin = np.random.randint(0, len(data)-batch_size)
            
            index_list = []
            input_example_list = []
            input_ids_list = []
            input_mask_list = []
            segment_ids_list = []
            pred_ids_list = []

            for i in range(imb_orgin, imb_orgin+batch_size):
                if len(input_ids_list) == batch_size:
                    break

                if len(data[i].text_a) == len(data[i].label):
                    single_input_ids,single_input_mask,single_segment_ids,single_pred_ids = convert_single_example(data[i], max_seq_length, tokenizer)
                    input_example_list.append(data[i])
                    input_ids_list.append(single_input_ids)
                    input_mask_list.append(single_input_mask)
                    segment_ids_list.append(single_segment_ids)
                    pred_ids_list.append(single_pred_ids)
            
                    index_list.append(i)
            #print(index_list)
            yield input_example_list, input_ids_list, input_mask_list, segment_ids_list, pred_ids_list
    else:
        num_sample = len(data)
        ind_start = 0

        while ind_start < num_sample:
            ind_end = min(ind_start+batch_size, num_sample)
            imb = np.arange(ind_start, ind_end)
            
            input_example_list = []
            input_ids_list = []
            input_mask_list = []
            segment_ids_list = []
            pred_ids_list = []
            
            for i in imb:
                if len(data[i].text_a) == len(data[i].label):
                    single_input_ids,single_input_mask,single_segment_ids,single_pred_ids = convert_single_example(data[i], max_seq_length, tokenizer)
                    input_example_list.append(data[i])
                    input_ids_list.append(single_input_ids)
                    input_mask_list.append(single_input_mask)
                    segment_ids_list.append(single_segment_ids)
                    pred_ids_list.append(single_pred_ids)
            
            ind_start += batch_size
            yield input_example_list, input_ids_list, input_mask_list, segment_ids_list, pred_ids_list

def create_examples(file_path):
    examples = []

    with open(file_path, 'r') as f:
        guid = 0
        text_a = []
        label = []
        for line in f:
            if line.strip() == '':
                examples.append(InputExample(guid=str(guid), text_a=text_a, label=label)) 
                text_a = []
                label = []
                guid += 1
            else:
                list_line = line.strip().split()
                text_a.append(list_line[0])
                if len(list_line) == 2:
                    label.append(list_line[1])
    
    return examples


