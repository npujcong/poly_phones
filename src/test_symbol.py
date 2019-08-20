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
import json
import numpy as np

from symbol import Symbol

def test_symbol(vocab_path, data_path, poly_dict_path):
    symbol = Symbol(vocab_path, poly_dict_path)
    with open(data_path, "r") as json_file:
        data = json.load(json_file)
    metadata=list(zip(data["features"], data["labels"]))
    for meta in metadata:
        input_data = np.asarray(symbol.feature_to_sequence(meta[0]), dtype = np.float32)
        target_data = np.asarray(symbol.label_to_sequence(meta[1]), dtype = np.float32)
        poly_mask = symbol.poly_mask(meta[0])
        print(meta[0])
        print(meta[1])
        print("input_data"+"="*50)
        print(input_data)
        print("target_data"+"="*50)
        print(target_data)
        print("poly_mask"+"="*50)
        print(poly_mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--vocab_path',
         default="/home/work_nfs3/jcong/workspace-3/POLY/blstm-poly/egs/poly_disambiguation/data/vocab.json"
    )
    parser.add_argument(
        '--data_path',
        default="/home/work_nfs3/jcong/workspace-3/POLY/blstm-poly/egs/poly_disambiguation/data/train.json"
    )
    parser.add_argument(
        '--poly_dict_path',
        default="/home/work_nfs3/jcong/workspace-3/POLY/blstm-poly/egs/poly_disambiguation/data/poly_pinyin_dict.json"
    )
    args = parser.parse_args()
    test_symbol(args.vocab_path, args.data_path, args.poly_dict_path)
