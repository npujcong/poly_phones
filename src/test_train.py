#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# Copyright ASLP@NPU. All Rights Reserved
#
# Licensed under the Apache License, Veresion 2.0(the "License");
# You may not use the file except in compliance with the Licese.
# You may obtain a copy of the License at
#
#   http://www.apache.org/license/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author npujcong@gmail.com(congjian)
# Date 2019/07/30 16:35:25
#
######################################################################

import os
import sys
import tensorflow as tf
import argparse
import numpy as np
from poly_model import Poly_Model
from dataset import DataFeeder
from symbol import  Symbol
import numpy as np

def restore_from_ckpt(sess, saver, save_dir):
    ckpt = tf.train.get_checkpoint_state(os.path.join(save_dir, "nnet"))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        return True
    else:
        tf.logging.fatal("checkpoint not found")
        return False

def test_decode(hparams):
    hp = hparams
    coord = tf.train.Coordinator()
    feeder = DataFeeder(coord, hp, False)
    symbol = Symbol(hp.vocab_path, hp.poly_dict_path)

    # construct model
    inputs, target_lengths, targets, poly_mask = feeder.dequeue()
    model = Poly_Model(hp, feeder.input_dim, feeder.num_class)
    model.initialize(inputs, target_lengths, targets, poly_mask)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        feeder.start_in_session(sess)
        sess.run(tf.global_variables_initializer())
        if not restore_from_ckpt(sess, saver, hp.save_dir): sys.exit(-1)
        test_acc = 0
        num_batchs = int(feeder.num_samples / hp.batch_size)
        tmp=0
        np.set_printoptions(precision=3)

        for i in range(num_batchs):
            # print(i)
            correct = 0
            total = 0
            model_pre, model_acc, model_outputs, model_targets, model_target_seq \
                = sess.run([model.pred, model.accuracy, model.outputs, model.targets, model.target_seq])
            print(model_pre)
            print(model_acc)
            print(model_outputs)
            print(model_targets)
            print(model_target_seq)
            # np.savetxt("{}.out".format(tmp), a[0], fmt='%1.4e')
            for pred_poly_seq in symbol.sequence_to_label(model_pre):
                print("\t".join(pred_poly_seq))
            for ta in symbol.sequence_to_label(model_target_seq):
                print("\t".join(ta))
            for pred_poly_seq, ta in zip(symbol.sequence_to_label(model_pre),symbol.sequence_to_label(model_target_seq)):
                if len(pred_poly_seq) != len(ta):
                    print("length not equal")
                    print("-----------------")
                    continue
                for p, t in zip(pred_poly_seq, ta):
                    if t == "-":
                        continue
                    if p == t:
                        correct += 1
                    total += 1
            print("Model Accuracy: {}".format(model_acc))
            print("Test  Accuracy New: {}".format(float(correct)/float(total)))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--decode',
        default=False,
        help="Flag indicating decoding or training.",
        action="store_true"
    )
    parser.add_argument(
        '--eval_size',
        type=int,
        default=0,
        help='eval_set size'
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=5.0,
        help='The max gradient normalization.'
    )
    parser.add_argument(
        '--queue_capacity',
        type=int,
        default=2,
        help='the FIFO queue capacity'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default="data/train.json",
        help='the FIFO queue capacity'
    )
    parser.add_argument(
        '--vocab_path',
        type=str,
        default="data/vocab.json",
        help='the FIFO queue capacity'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default="exp/",
        help='the FIFO queue capacity'
    )
    parser.add_argument(
        '--rnn_depth',
        type=int,
        default=2,
        help='Number of layers of rnn model.'
    )
    parser.add_argument(
        '--rnn_num_hidden',
        type=int,
        default=64,
        help='Number of hidden units to use.'
    )
    parser.add_argument(
        '--dnn_depth',
        type=int,
        default=1,
        help='Number of layers of rnn model.'
    )
    parser.add_argument(
        '--dnn_num_hidden',
        type=int,
        default=64,
        help='Number of hidden units to use.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Mini-batch size.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=500,
        help='Max number of epochs to run trainer totally.',
    )
    parser.add_argument(
        '--resume_training',
        default=False,
        help='Max number of epochs to run trainer totally.',
        action="store_true"
    )
    parser.add_argument(
        '--poly_dict_path',
        default="data/poly_pinyin_dict.json"
    )
    hp = parser.parse_args()
    test_decode(hp)
