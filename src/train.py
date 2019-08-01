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

import tensorflow as tf
import argparse
from model import poly_model
from dataset import datafeeder

def train_one_epoch():
    pass
def eval_one_epoch():
    pass

def main(hparams):
    hp = hparams
    model = poly_model(hparams)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hp.max_epoch):
            train_one_epoch()
        tf.logging.info("Epoch:{}".format(epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size")
    parser.add_argument("--learning_rate")
