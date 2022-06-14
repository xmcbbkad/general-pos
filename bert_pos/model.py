# -*- coding: utf-8 -*-

import tensorflow as tf
from bert import modeling
from bert import optimization

class BertPos(object):
    def __init__(self, hps):
        tf.logging.info("***Init BertPos-----------***")
        self.hps = hps
        self.mode = hps.mode
        self.max_seq_length = hps.max_seq_length
        self.num_labels = hps.num_labels
        self.lr = hps.lr
        self.init_checkpoint = hps.init_checkpoint

        self.bert_config = modeling.BertConfig.from_json_file(hps.bert_config_file)

        self.graph = tf.Graph()
        with self.graph.as_default():
            #self.global_step = tf.Variable(0, trainable = False)
            #self.global_step = tf.train.get_or_create_global_step()

            if self.mode == "train":
                self.build_graph()
                self.build_loss()
                self.setup_train()
                self.setup_summary()
            elif self.mode == "predict":
                self.build_graph()
            elif self.mode == "export":
                self.build_graph()

            self.init_fn()

    def build_graph(self):
        self.input_ids = tf.placeholder(tf.int32,[None,self.max_seq_length],name = 'input_ids')
        self.input_mask = tf.placeholder(tf.int32,[None,self.max_seq_length],name = 'input_mask')
        self.segment_ids = tf.placeholder(tf.int32,[None,self.max_seq_length],name = 'segment_ids')
        
        self.labels = tf.placeholder(tf.int32,[None,self.max_seq_length],name = 'labels')


        model = modeling.BertModel(
            config=self.bert_config,
            is_training=(self.mode == 'train'),
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)

        output_layer = model.get_sequence_output()
        
        with tf.name_scope('my_dense_layer'):
            if self.mode == 'train':
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            self.probabilities = tf.layers.Dense(self.num_labels, activation='softmax')(output_layer)
            print('probabilities:\t', self.probabilities.shape)
            self.predictions = tf.argmax(self.probabilities, axis=-1, output_type=tf.int32)
            print('prediction:\t', self.predictions.shape)
            
    def build_loss(self):
        with tf.name_scope('my_loss'):
            self.one_hot_labels = tf.one_hot(self.labels, depth=self.num_labels, dtype=tf.float32)
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.one_hot_labels * tf.log(self.probabilities), axis=-1))

    def setup_train(self):
        '''
        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(self.lr)
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        '''
        self.num_train_steps = self.hps.num_train_steps
        self.updates = optimization.create_optimizer(self.loss, self.lr, self.num_train_steps, int(self.num_train_steps*0.1), False)

    def setup_summary(self):
        tf.summary.scalar("loss", self.loss)
        self.summary_op = tf.summary.merge_all()

    def init_fn(self):
        tvars = tf.trainable_variables()
        
        if self.init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
            tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")
        
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

    def train_one_step(self, input_ids, input_mask, segment_ids, labels, sess):
        feed_dict = {}
        feed_dict[self.input_ids] = input_ids
        feed_dict[self.input_mask] = input_mask
        feed_dict[self.segment_ids] = segment_ids
        feed_dict[self.labels] = labels
        
        loss, _, summary = sess.run([self.loss, self.updates, self.summary_op], feed_dict=feed_dict)
        
        return loss, summary

    def predict(self, input_ids, input_mask, segment_ids, sess):
        feed_dict = {}
        feed_dict[self.input_ids] = input_ids
        feed_dict[self.input_mask] = input_mask
        feed_dict[self.segment_ids] = segment_ids
        
        predictions = sess.run(self.predictions, feed_dict=feed_dict)

        return predictions
