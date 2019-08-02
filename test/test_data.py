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

import argparse
import tensorflow as tf
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

from src.dataset import DataFeeder

def test_data(hp):
    coord = tf.train.Coordinator()
    feeder = DataFeeder(coord, hp)
    inputs, input_lengths, targets = feeder.dequeue()
    with tf.Session() as sess:
        feeder.start_in_session(sess)
        while(True):
            a, b, c = sess.run([inputs, input_lengths, targets])
            print("inputs shape: {}".format(a.shape))
            print("input_lengths shape: {}".format(b.shape))
            print("targets shape: {}".format(c.shape))

            print("inputs: {}".format(a))
            print("input_lengths : {}".format(b))
            print("targets: {}".format(c))

if __name__ == '__main__':
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
        '--queue_capacity',
        type=int,
        default=2,
        help='the FIFO queue capacity'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default="/home/work_nfs/jcong/workspace/blstm-chshan/egs/poly_disambiguation/data/train.txt",
        help='the FIFO queue capacity'
    )

    parser.add_argument(
        '--vocab_path',
        type=str,
        default="/home/work_nfs/jcong/workspace/blstm-chshan/egs/poly_disambiguation/data/vocab.json",
        help='the FIFO queue capacity'
    )

    parser.add_argument(
        '--input_dim',
        type=int,
        default=145,
        help='The dimension of inputs.'
    )
    parser.add_argument(
        '--num_class',
        type=int,
        default=75,
        help='The dimension of outputs.'
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
        default=30,
        help='Max number of epochs to run trainer totally.',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/',
        help='Directory of train, val and test data.'
    )
    hp = parser.parse_args()
    test_data(hp)