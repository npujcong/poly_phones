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
import json
import random
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
    
    def get(self):
        return {
            "value": self._value,
            "pos": self._pos,
            "pinyin" : self._pinyin,
            "ispoly" : self._ispoly,
            "poly_index" : self._poly_index
        }


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
    for sentence_index, line in enumerate(pos_lines):
        character_index = 0
        sentence = []
        for [value, _, pos] in [item.split("_") for item in line.strip().split(" ")]:
            if value not in PUNCTUATION:
                pinyin = pinyin_lines[sentence_index][character_index : character_index + len(value)]
                character_index += len(value)
                w = words(value, pos, pinyin)
            else:
                w = words(value, pos, None)
            sentence.append(w.get())
            print(w.get())
        print("\n")
        corpus.append(sentence)
    return corpus

def build_vocab_idx(dataset, vocab_path):
    features = dataset["features"]
    labels = dataset["labels"]
    # TODO 要不要把出现次数特别少的字加入进去？
    full_value_vocab = set(w[0] for sent in features for w in sent)
    word2idx = {}
    for word in full_value_vocab:
        word2idx[word] = len(word2idx)
    # TODO 注意 UNK
    full_pos_vocab = set(w[3] for sent in features for w in sent)
    pos2idx = {}
    for pos in full_pos_vocab:
        pos2idx[pos] = len(pos2idx)
    # Label 
    full_poly_vocab = set(poly for sent in labels for poly in sent)
    poly2idx = {}
    for poly in full_poly_vocab:
        poly2idx[poly] = len(poly2idx)
    
    seq2id = {
        "value_vocab": word2idx,
        "pos_vocab": pos2idx,
        "poly_vocab": poly2idx
    }
    json_str = json.dumps(seq2id, ensure_ascii=False, indent=2)
    with open(vocab_path, "w") as json_file:
        json_file.write(json_str)

# [word1, word2, word3...] 
# --> [(character, left_pos, right_pos, pos, is_poly), (), ()...]

def construct_dataset(corpus, train_dataset, test_dataset):
    features, labels = [], []
    for sentence in corpus:
        feat, lab = construct_sentence_feature(sentence)
        features.append(feat)
        labels.append(lab)
    # shuffle dataset
    tmp = list(zip(features, labels))
    random.shuffle(tmp)
    features, labels = zip(*tmp)
    dataset = {
        "features": features,
        "labels": labels
    }
    json_str = json.dumps(dataset, ensure_ascii=False, indent=2)
    with open(train_dataset, "w") as json_file:
        json_file.write(json_str)
    return dataset

def construct_sentence_feature(sentence):
    feature, label = [], []
    for i, word in enumerate(sentence):
        for character_index, character in enumerate(word["value"]):
            value = character
            pos = word["pos"]
            ispoly = word["ispoly"]
            poly_index = word["poly_index"]
            left_index = i - 1 if (i - 1) > 0 else i
            right_index = i + 1 if (i + 1) < len(sentence) else i
            left_neighbour_pos = sentence[left_index]["pos"]
            right_neighbour_pos = sentence[right_index]["pos"]
            if ispoly and character_index == poly_index:
                label.append(word["pinyin"][poly_index])
            else:
                label.append("-")
            feature_word = (value, left_neighbour_pos, right_neighbour_pos, pos, ispoly)
            feature.append(feature_word)
    return feature, label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinyin_txt", default="data/pinyin.txt")
    parser.add_argument("--pos_txt", default="data/pos.txt")
    parser.add_argument("--train", default="data/train.json")
    parser.add_argument("--test", default="data/test.json")
    parser.add_argument("--poly_dict", default="data/poly_dict")
    parser.add_argument("--vocab_path", default="data/vocab.json")

    args = parser.parse_args()
    with open(args.poly_dict, 'rb') as f_poly:
        POLY_DICT = pickle.load(f_poly)
    corpus = process_raw(args)
    dataset = construct_dataset(corpus, args.train, args.test)
    build_vocab_idx(dataset, args.vocab_path)
    # print(args.train)
    # with open(args.train, "r") as json_file:
    #     # json_file.write(json_str)
    #     data = json.load(json_file)
    #     print(data["features"])
    #     print(data["labels"])