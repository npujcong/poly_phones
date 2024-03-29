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
# Date 2019/07/26 11:02:19
#
######################################################################

from collections import defaultdict
import codecs
import json
import pickle
from pypinyin import style

def create_poly_dic():
    poly_dict = defaultdict(list)
    with codecs.open("polyphones.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for item in [x for x in lines if x != '\n']:
        words = item.strip().split()
        for p in words[1].split(','):
            tmp_p = style.convert(p, style=8, strict=False)
            # 注意：原始文件中轻声是没有音调的，但在我们的数据集中轻声是用5表示的
            if tmp_p[-1] not in ['1','2','3','4']:
                tmp_p += '5'
            poly_dict[words[-1]].append(tmp_p)
    json_str = json.dumps(poly_dict, ensure_ascii=False, indent=2)
    with open('polyphones.json', "w") as json_file:
        json_file.write(json_str)

def count_poly_words():
    poly_word_count=defaultdict(int)
    high_frequency_word = defaultdict(int)
    with open('polyphones.pickle', 'rb') as f:
        poly_dict = pickle.load(f)
    with codecs.open("data_poly_words.txt", "r", encoding='utf-8') as f:
        lines = f.readlines()
    for item in lines:
        for words in item.strip().split("\t")[1].split(" "):
            for word in words:
                if word in poly_dict.keys():
                    poly_word_count[word] += 1
    for item in sorted(poly_word_count.items(), key=lambda kv:(kv[1],kv[0]),reverse=True)[:75]:
        high_frequency_word[item[0]] = item[1]
    with open('high_frequency_word.pickle', 'wb') as f:
        pickle.dump(high_frequency_word, f)

if __name__ == '__main__':
    create_poly_dic()
    # count_poly_words()
    # with open('high_frequency_word.pickle', 'rb') as f:
    #     data = pickle.load(f)
    #     print(data)
