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
# Date 2019/07/30 16:41:41
#
######################################################################

import json
import numpy as np
import random
import tensorflow as tf
import threading
import traceback

from symbol import Symbol

_pad = 0

class DataFeeder(threading.Thread):
    '''Feeds batches of data into a queue on a background thread.'''

    def __init__(self, coordinator, hparams, shuffle):
        super(DataFeeder, self).__init__()
        self._coord = coordinator
        self._hparams = hparams
        hp = self._hparams
        self._offset = 0
        self._symbol = Symbol(hp.vocab_path, hp.poly_dict_path)
        self.input_dim = self._symbol.input_dim
        self.num_class = self._symbol.num_class
        self.shuffle = shuffle

        with open(hp.data_path, "r") as json_file:
            data = json.load(json_file)
            self._eval_feature =  data["features"][: hp.eval_size]
            self._eval_label = data["features"][: hp.eval_size]

            self._metadata=list(zip(data["features"][hp.eval_size:],
                data["labels"][hp.eval_size:]))

        self.num_samples = len(self._metadata)
        self._placeholders = [
            tf.placeholder(tf.float32, [None, None, self.input_dim], 'inputs'),
            tf.placeholder(tf.int32, [None], 'target_lengths'),
            tf.placeholder(tf.float32, [None, None, self.num_class], 'targets'),
            tf.placeholder(tf.float32, [None, None, self.num_class], 'poly_mask')
        ]

        # Create queue for buffering data:
        self.queue = tf.FIFOQueue(
            hp.queue_capacity,
            [tf.float32, tf.int32, tf.float32, tf.float32],
            name='input_queue')
        self._enqueue_op = self.queue.enqueue(self._placeholders)

    def start_in_session(self, session):
        self._session = session
        self.start()

    def run(self):
        try:
            while not self._coord.should_stop():
                self._enqueue_next_group()
        except Exception as e:
            traceback.print_exc()
            self._coord.request_stop(e)

    def dequeue(self):
        (inputs, target_lengths, targets, poly_mask) = self.queue.dequeue()
        inputs.set_shape(self._placeholders[0].shape)
        target_lengths.set_shape(self._placeholders[1].shape)
        targets.set_shape(self._placeholders[2].shape)
        poly_mask.set_shape(self._placeholders[3].shape)
        return inputs, target_lengths, targets, poly_mask

    def _enqueue_next_group(self):
        # Read a group of examples:
        batch_size = self._hparams.batch_size
        batches_per_group = self._hparams.queue_capacity
        examples = [
            self._get_next_example() 
            for i in range(batch_size * batches_per_group)
        ]
        # Local sorted for computational efficiency
        if self.shuffle:
            examples.sort(key=lambda x: x[-2])

        # Bucket examples based on similar output sequence length for efficiency:
        batches = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            feed_dict = dict(zip(self._placeholders, _prepare_batch(batch, self.shuffle)))
            self._session.run(self._enqueue_op, feed_dict=feed_dict)

    def _get_next_example(self):
        '''Loads a single example (input, mel_target, linear_target, cost) from disk'''
        if self._offset >= len(self._metadata):
            self._offset = 0
            if self.shuffle:
                random.shuffle(self._metadata)
        meta = self._metadata[self._offset]
        self._offset += 1
        # TODO
        input_data = np.asarray(self._symbol.feature_to_sequence(meta[0]), dtype=np.float32)
        target_data = np.asarray(self._symbol.label_to_sequence(meta[1]), dtype=np.float32)
        poly_mask = np.asarray(self._symbol.poly_mask(meta[0]), dtype = np.float32)
        return (input_data, target_data, input_data.shape[0], poly_mask)

def _prepare_batch(batch, shuffle):
    if shuffle:
        random.shuffle(batch)
    inputs = _prepare([x[0] for x in batch])
    targets = _prepare([x[1] for x in batch])
    targets_lengths = np.asarray([x[1].shape[0] for x in batch], dtype=np.int32)
    poly_mask = _prepare([x[3] for x in batch])
    return (inputs, targets_lengths, targets, poly_mask)

# batch & pad
def _prepare(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_input(x, max_len) for x in inputs])

def _pad_input(x, length):
    return np.pad(
        x, [(0, length - x.shape[0]), (0, 0)],
        mode='constant',
        constant_values=_pad)