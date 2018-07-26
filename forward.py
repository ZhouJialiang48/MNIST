#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/26 09:15
# @Author  : zhoujl
# @Site    : 
# @File    : forward.py
# @Software: PyCharm
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER_1_NODE = 500

def forward(x, regularizer):
    w1 = get_weight(shape=[INPUT_NODE, LAYER_1_NODE], regularizer=regularizer)
    b1 = get_bias(shape=[LAYER_1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight(shape=[LAYER_1_NODE, OUTPUT_NODE], regularizer=regularizer)
    b2 = get_bias(shape=[OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2

    return y


def get_weight(shape, regularizer=None):
    w = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
    if regularizer:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b
