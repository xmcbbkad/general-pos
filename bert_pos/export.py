# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import json

from bert import tokenization
from model import BertPos

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('mode', 'export', 'Must be one of train/predict/export')
flags.DEFINE_string('log_root', 'log', 'Root directory for all logging.')
flags.DEFINE_string('export_dir', 'output_model', "The output directory where the model checkpoints will be written.")
flags.DEFINE_string('model_version', '00000', 'Version for the exported model')


def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not os.path.exists(FLAGS.export_dir):
        os.makedirs(FLAGS.export_dir)

    export_path = os.path.join(FLAGS.export_dir, FLAGS.model_version)

    hparams_path = os.path.join(FLAGS.log_root, 'hparams.json')
    with open(hparams_path, 'r') as result_file:
        h_json = json.load(result_file)
        h_json = json.loads(h_json)
        h_json['mode'] = FLAGS.mode

        hparams = tf.contrib.training.HParams()
        for key,value in h_json.items():
            hparams.add_hparam(key,value)

    model = BertPos(hps = hparams)

    with model.graph.as_default():
        with tf.Session() as sess:
            tf.logging.info('Start to load the latest checkpoint...') 
            saver = tf.train.Saver()
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.log_root))
            tf.logging.info('Checkpoint loaded...')

            builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        
            tfs_input_ids = tf.saved_model.utils.build_tensor_info(model.input_ids)
            tfs_input_mask = tf.saved_model.utils.build_tensor_info(model.input_mask)
            tfs_segment_ids = tf.saved_model.utils.build_tensor_info(model.segment_ids)

            outputs = tf.saved_model.utils.build_tensor_info(model.predictions)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        'input_ids': tfs_input_ids,
                        'input_mask': tfs_input_mask,
                        'segment_ids': tfs_segment_ids
                    },
                    outputs={
                        'outputs': outputs
                    },
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
            )
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict':prediction_signature,
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:prediction_signature
                        },
                legacy_init_op=legacy_init_op
            )

            builder.save()
            tf.logging.info('Done exporting!')


if __name__ == "__main__":
    tf.app.run()

