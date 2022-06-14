# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import util.ner_metrics

import tensorflow as tf
import numpy as np
import os
import pickle
import json
from tqdm import tqdm

from bert import tokenization
from model import BertPos
import common

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('mode', 'train', 'Must be one of train/decode')
flags.DEFINE_string('log_root', 'log', 'Root directory for all logging.')
flags.DEFINE_string('bert_config_file', 'chinese_L-12_H-768_A-12/bert_config.json', 'Path of the bert_config file')
flags.DEFINE_string('init_checkpoint', 'chinese_L-12_H-768_A-12/bert_model.ckpt', 'Path of the init_checkpoint')
flags.DEFINE_string('vocab_path', 'chinese_L-12_H-768_A-12/vocab.txt', 'Path of the vocab file')
flags.DEFINE_string('train_file_path', '', 'Path of the train file')
flags.DEFINE_string('dev_file_path', '', 'Path of the dev file')

flags.DEFINE_integer('n_epoch', 20, 'Number of epoch to train the model')
flags.DEFINE_integer('batch_size', 32, 'Minibatch size')
flags.DEFINE_integer('max_seq_length', 128, 'Max length of the sequence')
flags.DEFINE_float('lr', 2e-5, 'Learning rate')
flags.DEFINE_integer('max_to_keep', 4, 'Number of checkpoints to keep')
flags.DEFINE_integer('steps_per_eval', 500, 'The frequency to evaluate the model')


def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    
    if not os.path.exists(FLAGS.log_root):
        os.makedirs(FLAGS.log_root)

    hparams = tf.contrib.training.HParams()
    for ind,item in enumerate(FLAGS):
        attr = FLAGS.__getattr__(item)
        print(item + ":" + str(attr) + "   type:"+ str(type(attr)))
        hparams.add_hparam(name = item,value = attr)
    hparams.add_hparam(name = 'num_labels', value = len(common.LABEL_MAP))
    s = hparams.to_json()
    hparams_path = os.path.join(FLAGS.log_root,'hparams.json')
    with open(hparams_path, 'w') as result_file:
        json.dump(s, result_file)

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_path, do_lower_case=False)

    print('Start to load the training data...')
    train_data = common.create_examples(FLAGS.train_file_path)
    print('Training data loaded')
    train_data_batch = common.generate_batch(train_data, FLAGS.batch_size, FLAGS.max_seq_length, tokenizer, True) 

    num_train_steps = int(len(train_data) / FLAGS.batch_size * FLAGS.n_epoch)
    hparams.add_hparam(name="num_train_steps", value = num_train_steps)

    print('Start to load the dev data...')
    dev_data = common.create_examples(FLAGS.dev_file_path)
    print('Dev data loaded')

    model = BertPos(hps = hparams) 

    with model.graph.as_default():
        sv = tf.train.Supervisor(logdir=FLAGS.log_root,
                                 saver = tf.train.Saver(max_to_keep = FLAGS.max_to_keep),
                                 summary_op=None,
                                 save_model_secs=360,
                                 global_step = tf.train.get_or_create_global_step(),
                                 init_op=tf.global_variables_initializer()
                                 ) # Do not run the summary service

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True

        with sv.managed_session(config = config) as sess:
            n_iter_per_epoch = len(train_data) // FLAGS.batch_size
            epoch = 0
            print('number of iterations per epoch: {}'.format(n_iter_per_epoch))
            print('start training...')
            for _ in range(FLAGS.n_epoch):
                epoch += 1
                avg_loss = 0.0
                print("----- Epoch {}/{} -----".format(epoch, FLAGS.n_epoch))

                for t in tqdm(range(1, n_iter_per_epoch + 1)):
                    input_raw, input_ids_list, input_mask_list, segment_ids_list, pred_ids_list = next(train_data_batch)
                    input_ids_list = np.asarray(input_ids_list,dtype = np.int32)
                    input_mask_list = np.asarray(input_mask_list,dtype = np.int32)
                    segment_ids_list = np.asarray(segment_ids_list,dtype = np.int32)
                    pred_ids_list = np.asarray(pred_ids_list,dtype = np.int32)

                    loss, summary = model.train_one_step(input_ids_list, input_mask_list, segment_ids_list, pred_ids_list, sess)
                    avg_loss += loss

                    if t%2 == 0:
                        sv.summary_computed(sess, summary)
                    
                    global_step = sess.run(tf.train.get_or_create_global_step())
                    if global_step>0 and global_step % FLAGS.steps_per_eval==0:
                        dev_data_batch = common.generate_batch(dev_data, FLAGS.batch_size, FLAGS.max_seq_length, tokenizer, False) 
                        print('step {} , start to evaluate...'.format(global_step))
                        
                        true_label_all = np.array([])
                        pred_label_all = np.array([])
                        for dev_input_raw, dev_input_ids_list, dev_input_mask_list, dev_segment_ids_list, dev_pred_ids_list in dev_data_batch:
                            dev_input_ids_list = np.asarray(dev_input_ids_list,dtype = np.int32)
                            dev_input_mask_list = np.asarray(dev_input_mask_list,dtype = np.int32)
                            dev_segment_ids_list = np.asarray(dev_segment_ids_list,dtype = np.int32)
                            dev_pred_ids_list = np.asarray(dev_pred_ids_list,dtype = np.int32)
                            
                            predictions = model.predict(dev_input_ids_list, dev_input_mask_list, dev_segment_ids_list, sess) 
                            pred_label_all = np.append(pred_label_all, predictions.flatten())
                            true_label_all = np.append(true_label_all, dev_pred_ids_list.flatten())
                            
                        util.ner_metrics.my_F1(true_label_all, pred_label_all, is_str=False)                                                

                avg_loss /= n_iter_per_epoch
                print('the avg_loss is {}'.format(avg_loss))

if __name__ == "__main__":
    tf.app.run()
