#!/bin/bash
# Copyright 2016 ASLP@NPU.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: npujcong@gmail.com (congjian)

stage=1
current_working_dir=$(pwd)
pro_dir=$(dirname $(dirname $current_working_dir))

thulac_dir=$pro_dir/tools/THULAC/
model_dir=$pro_dir/models
raw=$current_working_dir/raw
data=$current_working_dir/data

# set -euo pipefail
[ ! -e $data ] && mkdir -p $data

# step 1: word segment and pos_tag
if [ $stage -le 0 ]; then
  awk -F'\t' '{print $1}' $raw/raw.utf8 > $raw/text.utf8
  awk -F'\t' '{print $2}' $raw/raw.utf8 > $raw/pinyin.utf8
  $thulac_dir/build/thulac \
    -model_dir $thulac_dir/models \
    -input $raw/text.utf8 \
    -output $raw/pos.utf8
fi

# step 2: preprocess data and generate train.txt、test.txt
if [ $stage -le 1 ]; then
  $pro_dir/src/preprocess.py \
    --pinyin_txt $raw/pinyin.utf8 \
    --pos_txt $raw/pos.utf8 \
    --train $data/train.txt \
    --test $data/test.txt \
    --poly_dict $pro_dir/data/high_frequency_word.pickle
fi
