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

stage=$1
current_working_dir=$(pwd)
pro_dir=$(dirname $(dirname $current_working_dir))

thulac_dir=$pro_dir/tools/THULAC/
model_dir=$pro_dir/models
raw=$current_working_dir/raw
data=$current_working_dir/data
exp=$current_working_dir/exp

# set -euo pipefail
[ ! -e $data ] && mkdir -p $data
[ ! -e $exp ] && mkdir -p $exp

# step 0: word segment and pos_tag
if [ $stage -le 0 ]; then
  for item in test train
  do
    awk -F'\t' '{print $1}' $raw/$item-raw.utf8 > $data/$item-text.utf8
    awk -F'\t' '{print $2}' $raw/$item-raw.utf8 > $data/$item-pinyin.utf8
    $thulac_dir/build/thulac \
      -model_dir $thulac_dir/models \
      -input $data/$item-text.utf8 \
      -output $data/$item-pos.utf8
  done
fi

# step 1: preprocess data and generate train.txt、test.txt
if [ $stage -le 1 ]; then
  $pro_dir/src/preprocess.py \
    --train_pinyin_txt $data/train-pinyin.utf8 \
    --train_pos_txt $data/train-pos.utf8 \
    --test_pinyin_txt $data/test-pinyin.utf8 \
    --test_pos_txt $data/test-pos.utf8 \
    --train $data/train.json \
    --test $data/test.json \
    --poly_dict $raw/high_frequency_word.pickle \
    --vocab_path $data/vocab.json \
    --poly_pinyin_dict $data/poly_pinyin_dict.json
fi

# step 2: train model
if [ $stage -le 2 ]; then
  CUDA_VISIBLE_DEVICES=2 python $pro_dir/src/train.py \
    --dnn_depth 1 \
    --dnn_num_hidden 64 \
    --rnn_depth 2 \
    --rnn_num_hidden 256 \
    --batch_size 1 \
    --learning_rate 0.001 \
    --max_epochs 100 \
    --data_path $data/test.json \
    --vocab_path $data/vocab.json \
    --save_dir $exp \
    --poly_dict_path $data/poly_pinyin_dict.json
fi

# step 3: test model
if [ $stage -le 3 ]; then
  CUDA_VISIBLE_DEVICES= python $pro_dir/src/train.py \
    --decode \
    --dnn_depth 1 \
    --dnn_num_hidden 64 \
    --rnn_depth 2 \
    --rnn_num_hidden 256 \
    --batch_size 1 \
    --learning_rate 0.001 \
    --max_epochs 100 \
    --data_path $data/test.json \
    --vocab_path $data/vocab.json \
    --save_dir $exp \
    --poly_dict_path $data/poly_pinyin_dict.json
fi

# step 4: test model decode
if [ $stage -le 4 ]; then
  CUDA_VISIBLE_DEVICES= python $pro_dir/src/test_train.py \
    --decode \
    --dnn_depth 1 \
    --dnn_num_hidden 64 \
    --rnn_depth 2 \
    --rnn_num_hidden 256 \
    --batch_size 2 \
    --learning_rate 0.001 \
    --max_epochs 100 \
    --data_path $data/test.json \
    --vocab_path $data/vocab.json \
    --save_dir $exp \
    --poly_dict_path $data/poly_pinyin_dict.json
fi
