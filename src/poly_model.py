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
# Date 2019/07/30 16:34:11
#
######################################################################

import tensorflow as tf

class Poly_Model():
    def __init__(self, hparams, input_dim, num_class):
        self._hparams = hparams
        self._input_dim = input_dim
        self._num_class = num_class

    def initialize(self, inputs, target_lengths, targets, poly_mask):
        hp = self._hparams
        outputs = inputs
        for i in range(hp.dnn_depth):
            outputs = tf.layers.dense(outputs, hp.dnn_num_hidden, activation=tf.nn.sigmoid,
                name="dnn_{}".format(i))
        for i in range(hp.rnn_depth):
            cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=hp.rnn_num_hidden,
                name="fw_{}".format(i))
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=hp.rnn_num_hidden,
                name="bw_{}".format(i))
            # inputs [batch_size, T, feats_dim]
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                outputs, dtype=tf.float32)
            # A tuple (output_fw, output_bw) [batch_size, max_time, output_size]
            outputs = tf.concat([output_fw, output_bw], axis=2) #[batch_size, max_time, output_size * 2]
        outputs = tf.layers.dense(outputs, self._num_class, activation=None)

        self.inputs = inputs
        self.outputs = tf.nn.softmax(outputs)
        self.targets = targets
        self.target_lengths = target_lengths
        self.mask = tf.cast(
            tf.sequence_mask(self.target_lengths, tf.shape(self.outputs)[1]), tf.float32)
        self.target_seq = tf.argmax(self.targets, 2)
        self.poly_mask = poly_mask

        # accuracy & pred
        pred = tf.argmax(self.outputs  * self.poly_mask, 2)
        poly_index = tf.cast(tf.cast(self.target_seq, dtype=tf.bool), tf.float32)
        correct_pred = tf.cast(tf.equal(pred, self.target_seq), tf.float32) * poly_index 
        accuracy = tf.reduce_sum(correct_pred) / tf.reduce_sum(poly_index)

        self.pred = pred
        self.accuracy = accuracy
        self.correct_pred = correct_pred

    def add_loss(self):
        # Mask the logits sequence
        # [batch_size, T, 1]
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits = self.outputs, labels = self.targets)
        # loss: [batch_size, T]
        # mask: [batch_size, T]
        loss *= self.mask # [batch_size, T]
        # loss *= tf.cast(self.poly_mask, tf.float32)
        loss = tf.reduce_mean(loss)
        self.loss = loss