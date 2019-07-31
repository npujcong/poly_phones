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
# Date 2019/07/30 16:44:54
#
######################################################################
import argparse
import pickle

POLY_DICT={}
PUNCTUATION = [u'。', u'，', u'、', u'；', u'：', u'？', u'！', u'…',
               u'、', u'〃', u'‘', u'“', u'”', u'∶', u'＂', u'＇',
               u'｀', u'｜', u'〔', u'〕', u'〈', u'〉', u'《', u'》',
               u'「', u'」', u'『', u'』', u'．', u'〖', u'〗', u'【',
               u'】', u'（', u'）', u'［', u'］', u'｛', u'｝', u'.',
               u',', u'!', u' ', u'?', u'(', u')', u':', u"'", u'\"',u';',
               u'<', u'>', u'［' , u'］' , u'№' , u';' , u'『', u'∶', u'[', u']', u'{', u'}', u'，',u'\\', u'\n',u'',u'_']

class words:
    def __init__(self, value, pos, pinyin):
        self._value = value
        self._pos = pos
        self._pinyin = pinyin
        self._ispoly = self._is_poly(value)   # 多音字或者多音词
    
    def _is_poly(self, value):
        for i, item in enumerate(value):
            if item in POLY_DICT.keys():
                self._poly_index = i
                return True
        self._poly_index = None
        return False

    @property
    def value(self):
        return self._value
    
    @property
    def pos(self):
        return self._pos

    @property
    def pinyin(self):
        return self._pinyin
    
    @property
    def ispoly(self):
        return self._is_poly
    
    @property
    def poly_index(self):
        return self._poly_index

"""
return : corpus: [sentence1, sentence2]
         sentence: [word1, word2, word3...]
"""
def process_raw(args):
    with open(args.pinyin_txt, 'r') as f_pinyin:
        pinyin_lines = [line.strip().split(" ") for line in f_pinyin.readlines()]
    with open(args.pos_txt, 'r') as f_pos:
        pos_lines = f_pos.readlines()
    sentence, corpus = [],[]
    sentence_index = 0
    character_index = 0
    for line in pos_lines:
        if line in ['\n','\r\n']:
            corpus.append(sentence)
            for item in sentence:
                print(item.value, item.pos, item.pinyin)
            print("\n")
            character_index = 0
            sentence_index += 1
            sentence = []
        else:
            [value, _, pos] = line.strip().split('\t')
            if value not in PUNCTUATION:
                pinyin = pinyin_lines[sentence_index][character_index : character_index + len(value)]
                character_index += len(value)
                w = words(value, pos, pinyin)
            else:
                w = words(value, pos, None)
        sentence.append(w)

def construc_feature(corpus, train_path, test_path):
    context = 1
    for sentence in corpus:
        for i, word in enumerate(sentence):
            if word.ispoly:
                



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinyin_txt", default="data/pinyin.txt")
    parser.add_argument("--pos_txt", default="data/pos.txt")
    parser.add_argument("--train", default="data/train.txt")
    parser.add_argument("--test", default="data/test.txt")
    parser.add_argument("--poly_dict", default="data/poly_dict")
    args = parser.parse_args()
    with open(args.poly_dict, 'rb') as f_poly:
        POLY_DICT = pickle.load(f_poly)
    corpus = process_raw(args)
    construc_feature(corpus, args.train, args.test)