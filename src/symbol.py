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
    def __init__(self, vocab_path):
        with open(vocab_path, "r") as json_file:
            data = json.load(json_file)
            self._value_vocab = data["value_vocab"]
            self._pos_vocab = data["pos_vocab"]
            self._poly_vocab = data["poly_vocab"]
            self.input_dim = len(self._value_vocab.keys()) + 3 * len(self._pos_vocab.keys()) + 2
            self.num_class = len(self._poly_vocab.keys())

    def feature_to_sequence(self, feat):
        character = [self._value_vocab[x[0]] for x in feat]
        left_pos = [self._pos_vocab[x[1]] for x in feat]
        right_pos = [self._pos_vocab[x[2]] for x in feat]
        pos = [self._pos_vocab[x[3]] for x in feat]
        is_poly = [1 if x[4] else 0 for x in feat]
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
            self.onehot(is_poly, 2)), axis=1)

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
    def onehot(self, data, dim):
        return np.eye(dim)[data]