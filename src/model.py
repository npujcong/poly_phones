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

class poly_model():
    def __init__(self, hparams):
        self._hparams = hparams
    
    def initialize(self, inputs, input_lengths, targets):
        hp = self._hparams
        outputs = inputs
        for _ in range(hp.lstm_layers):
            cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=hp.lstm_size)
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=np.lstm_size)
            # inputs [batch_size, T, feats_dim]
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, outputs)
            # A tuple (output_fw, output_bw) [batch_size, max_time, output_size]
            outputs = tf.concat([output_fw, output_bw], axis=2) #[batch_size, max_time, output_size * 2]
        outputs = tf.layer.Dense(outputs, hp.num_class, activation = tf.nn.relu)

        self.inputs = inputs
        self.outputs = outputs
        self.targets = targets
        self.input_lengths = input_lengths

    def add_loss(self):
        # Mask the logits sequence
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits = pred, labels = y))
        self.loss = loss