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

import tensorflow as tf
import argparse
from poly_model import Poly_Model
from dataset import DataFeeder

def train_one_epoch(sess, train_step, train_loss, train_accuracy, 
    global_step, batchs_per_epoch):
    tr_loss = 0
    tr_acc = 0
    for i in range(batchs_per_epoch):
        _, loss, acc = sess.run([train_step, train_loss, train_accuracy])
        tr_loss += loss
        tr_acc += acc
    tr_loss /= float(batchs_per_epoch)
    tr_acc /= float(batchs_per_epoch)
    return tr_loss, tr_acc

def train(hparams):
    hp = hparams
    # data_feeder
    coord = tf.train.Coordinator()
    feeder = DataFeeder(coord, hp)
    numbatchs_per_epoch = int(feeder.num_samples / hp.batch_size)
    
    # construct model
    inputs, target_lengths, targets = feeder.dequeue()
    model = Poly_Model(hp, feeder.input_dim, feeder.num_class)
    model.initialize(inputs, target_lengths, targets)
    model.add_loss()
    loss = model.loss
    train_accuracy = model.compute_accuracy()

    # loss & optimizer
    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.int64,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])
    trainable_variables = tf.trainable_variables()
    print(trainable_variables)
    learning_rate = tf.get_variable("learning_rate", shape=[],dtype=tf.float32,
        initializer=tf.constant_initializer(hp.learning_rate))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(loss, trainable_variables), hp.max_grad_norm)
    train_step = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=global_step)

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        feeder.start_in_session(sess)
        sess.run(tf.global_variables_initializer())
        for epoch in range(hp.max_epochs):
            tr_loss, tr_acc = train_one_epoch(sess, train_step, loss, train_accuracy,
                global_step, numbatchs_per_epoch)
            tf.logging.info("Epoch:{} TRIAIN LOSS: {} TRAIN ACCURACY: {}".format(epoch, tr_loss, tr_acc))

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
        default=500,
        help='Max number of epochs to run trainer totally.',
    )
    hp = parser.parse_args()
    train(hp)
