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

def train_one_epoch(sess, train_step, train_loss, train_accuracy,
    merged, summary_writer, global_step, batchs_per_epoch):
    tr_loss = 0
    tr_acc = 0
    for step in range(batchs_per_epoch):
        # wirte summary every 50 step except step==0
        if step % 1000 == 999:
            _, loss, acc, summary, step = sess.run([train_step, train_loss,
                train_accuracy, merged, global_step])
            summary_writer.add_summary(summary, step)
            tf.logging.info("Writing Summary At Step {}".format(step))
        else:
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
    feeder = DataFeeder(coord, hp, True)
    numbatchs_per_epoch = int(feeder.num_samples / hp.batch_size)
    tf.logging.info("Num Batchs Per Epoch : {}".format(numbatchs_per_epoch))


    # construct model
    inputs, target_lengths, targets, poly_mask = feeder.dequeue()
    model = Poly_Model(hp, feeder.input_dim, feeder.num_class)
    model.initialize(inputs, target_lengths, targets, poly_mask)
    model.add_loss()
    loss = model.loss
    train_accuracy = model.accuracy


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
    learning_rate = tf.train.exponential_decay(hp.learning_rate, global_step,
        numbatchs_per_epoch, 0.8, staircase=True)
    # learning_rate = tf.get_variable("learning_rate", shape=[],dtype=tf.float32,
    #     initializer=tf.constant_initializer(hp.learning_rate))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(loss, trainable_variables), hp.max_grad_norm)
    train_step = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=global_step)

    # summary
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("learning_rate", learning_rate)
    merge_all = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=hp.max_epochs)
    config = tf.ConfigProto()

    with tf.Session(config=config) as sess:
        feeder.start_in_session(sess)
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(os.path.join(hp.save_dir, "nnet"), sess.graph)
        if hp.resume_training:
            restore_from_ckpt(sess, saver, hp.save_dir)
        for epoch in range(hp.max_epochs):
            tr_loss, tr_acc = train_one_epoch(sess, train_step, loss, train_accuracy,
                merge_all, summary_writer, global_step, numbatchs_per_epoch)
            tf.logging.info("Epoch:{} TRIAIN LOSS: {} TRAIN ACCURACY: {}".format(epoch, tr_loss, tr_acc))
            checkpoint_path = os.path.join(hp.save_dir, "nnet", "Epoch-{}-ACC-{}".format(epoch, tr_acc))
            saver.save(sess, checkpoint_path)
            tf.logging.info("Saving Checkpint At {}".format(checkpoint_path))

def decode(hparams):
    hp = hparams
    # data_feeder
    coord = tf.train.Coordinator()
    feeder = DataFeeder(coord, hp, False)
    symbol = Symbol(hp.vocab_path, hp.poly_dict_path)

    # construct model
    inputs, target_lengths, targets, poly_mask = feeder.dequeue()
    model = Poly_Model(hp, feeder.input_dim, feeder.num_class)
    model.initialize(inputs, target_lengths, targets, poly_mask)
    test_accuracy = model.accuracy
    predict_seq = model.pred

    saver = tf.train.Saver()
    with tf.Session() as sess:
        feeder.start_in_session(sess)
        sess.run(tf.global_variables_initializer())
        if not restore_from_ckpt(sess, saver, hp.save_dir): sys.exit(-1)
        test_acc = 0
        num_batchs = int(feeder.num_samples / hp.batch_size)
        tmp=0
        correct = 0
        total = 0
        for i in range(num_batchs):
            # print(i)
            pred, acc, a, c, tag = sess.run([predict_seq, test_accuracy, model.outputs, targets, model.target_seq])
            # print(pred)
            # print(a)
            # print(b)
            # print(c)
            # print("Pred Seq: {}".format(pred))
            # print(a)
            # np.savetxt("{}.out".format(tmp), a[0], fmt='%1.4e')
            # print(pred)
            for pred_poly_seq in symbol.sequence_to_label(pred):
                print("\t".join(pred_poly_seq))
            for ta in symbol.sequence_to_label(tag):
                print("\t".join(ta))
            for pred_poly_seq, ta in zip(symbol.sequence_to_label(pred),symbol.sequence_to_label(tag)):
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
            test_acc += acc
            tmp += 1
        test_acc /= float(num_batchs)
        tf.logging.info("Test Accuracy : {}".format(test_acc))
        tf.logging.info("Test Accuracy New: {}".format(float(correct)/float(total)))


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
        default="/home/work_nfs/jcong/workspace/blstm-chshan/egs/poly_disambiguation/data/train.json",
        help='the FIFO queue capacity'
    )
    parser.add_argument(
        '--vocab_path',
        type=str,
        default="/home/work_nfs/jcong/workspace/blstm-chshan/egs/poly_disambiguation/data/vocab.json",
        help='the FIFO queue capacity'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default="/home/work_nfs/jcong/workspace/blstm-chshan/egs/poly_disambiguation/exp/",
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
        default=0,
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
    if hp.decode:
        decode(hp)
    else:
        train(hp)
