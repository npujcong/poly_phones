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
    
    def initialize(self, inputs, target_lengths, targets):
        hp = self._hparams
        outputs = inputs
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
        outputs = tf.layers.dense(outputs, self._num_class, activation=tf.nn.relu)

        self.inputs = inputs
        self.outputs = outputs
        self.targets = targets
        self.target_lengths = target_lengths

    def add_loss(self):
        # Mask the logits sequence
        # [batch_size, T, 1]
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits = self.outputs, labels = self.targets)
        mask = tf.cast(
            tf.sequence_mask(self.target_lengths, tf.shape(self.outputs)[1]), tf.float32)
        loss *= mask
        loss = tf.reduce_mean(loss)
        self.loss = loss

    def compute_accuracy(self):
        correct_pred = tf.equal(
            tf.argmax(self.outputs, 1), tf.argmax(self.targets, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy