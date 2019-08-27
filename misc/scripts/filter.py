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
# Date 2019/08/23 17:30:07
#
######################################################################

from zhon.hanzi import punctuation
import string
import re
import codecs
import argparse

PUNCTUATION = list(punctuation + string.punctuation)
special_symbol=set()

def is_chinese(uchar):
    if uchar >= '\u4e00' and uchar<='\u9fa5':
        return True
    else:
        return False

def process_line(line):
    new_line = ""
    for item in line:
        if is_chinese(item) or item in PUNCTUATION:
            new_line += item
        else:
            special_symbol.add(item)
    return new_line

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="raw/test-raw.utf8")
    parser.add_argument("--output", default="data/test-raw.utf8")
    args = parser.parse_args()

    with codecs.open(args.input, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    fout = codecs.open(args.output, 'w', encoding='utf-8')

    for line in lines:
        if len(line.strip().split("\t")) != 2:
            continue
        sentence, pinyin = line.strip().split("\t")
        pinyin = re.sub(" pau0", "", pinyin)
        pinyin = re.sub(" +", " ", pinyin)
        fout.write(process_line(sentence) + "\t" + pinyin.strip() + "\n")

    print("="*10)
    for item in special_symbol:
        print(item)
    print(PUNCTUATION)
