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

import numpy as np
import json

class Symbol():
    def __init__(self, vocab_path, poly_dict_path):
        with open(poly_dict_path, 'r') as f_poly:
            self._poly_dict = json.load(f_poly)

        with open(vocab_path, "r") as json_file:
            data = json.load(json_file)
            self._value_vocab = data["value_vocab"]
            self._pos_vocab = data["pos_vocab"]
            self._poly_vocab = data["poly_vocab"]
            self._poly_word_vocab = data["poly_word2idx"]
            self._poly_vocab_reverse = {}
            for (key, value) in self._poly_vocab.items():
                self._poly_vocab_reverse[value] = key
            self.input_dim = len(self._value_vocab.keys()) + 3 * len(self._pos_vocab.keys()) + len(self._poly_word_vocab.keys())
            self.num_class = len(self._poly_vocab.keys())

    def feature_to_sequence(self, feat):
        character = [self._value_vocab[x[0]] for x in feat]
        left_pos = [self._pos_vocab[x[1]] for x in feat]
        right_pos = [self._pos_vocab[x[2]] for x in feat]
        pos = [self._pos_vocab[x[3]] for x in feat]
        is_poly = [self._poly_word_vocab[x[0]] if x[4] else self._poly_word_vocab["NOPOLY"] for x in feat]

        # print("feats", feat)
        # print("character", character)
        # print("left", left_pos)
        # print("right", right_pos)
        # print("pos", pos)
        # print("is_poly", is_poly)
        return  np.concatenate(
            (self.onehot(character, len(self._value_vocab.keys())),
            self.onehot(left_pos, len(self._pos_vocab.keys())),
            self.onehot(right_pos, len(self._pos_vocab.keys())),
            self.onehot(pos, len(self._pos_vocab.keys())),
            self.onehot(is_poly, len(self._poly_word_vocab.keys()))), axis=1)

    def poly_mask(self, feat):
        # print(feat)
        poly_mask = np.zeros([len(feat), self.num_class])
        for row, [character, _, _, _, is_poly] in enumerate(feat):
            if is_poly:
                # print(self._poly_dict[character])
                for poly in self._poly_dict[character]:
                    poly_mask[row][self._poly_vocab[poly]] = 1
                    # print(self._poly_vocab[poly])
            else:
                poly_mask[row][self._poly_vocab["-"]] = 1
        # print("POLY MASK:{}".format(np.argmax(poly_mask, 1)))
        return poly_mask

    def label_to_sequence(self, label):
        lab = [self._poly_vocab[x] for x in label]
        lab_onehot = self.onehot(lab, len(self._poly_vocab.keys()))
        return lab_onehot
    """
        data: [1, 3, 4]
        return:
        [[0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1]]

    """
    def sequence_to_label(self, sequence):
        batch_label = []
        for i in range(sequence.shape[0]):
            label = [self._poly_vocab_reverse[x] for x in sequence[i]]
            batch_label.append(label)
        return batch_label

    def onehot(self, data, dim):
        return np.eye(dim)[data]
