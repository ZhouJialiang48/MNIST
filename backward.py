#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/26 09:29
# @Author  : zhoujl
# @Site    : 
# @File    : backward.py
# @Software: PyCharm
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward

STEPS = 30000
LOG_CYCLE = 1000
BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.0001
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'mnist_model'


def backward(mnist):
    x = tf.placeholder(tf.float32, shape=[None, forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, shape=[None, forward.OUTPUT_NODE])
    y = forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    # sparse_softmax_cross_entropy_with_logits方法，
    # logits.shape=(BATCH_SIZE, 10), labels.shape=(BATCH_SIZE)，且labels必须为int
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        # 训练集的样本数量
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )

    # 此处global_step真正成为全局计数器
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    # 等待train_step和ema_op操作结束之后， 再进行下一操作
    # 此处下一步无实际操作，仅将两者重新命名
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_val = sess.run([train_op, loss], feed_dict={x: xs, y_: ys})
            if i % LOG_CYCLE == 0:
                print('Iter {}, loss is {}'.format(i, loss_val))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


if __name__ == '__main__':
    mnist = input_data.read_data_sets('./data/', one_hot=True)
    backward(mnist)

