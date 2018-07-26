#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/26 10:31
# @Author  : zhoujl
# @Site    : 
# @File    : evaluation.py
# @Software: PyCharm
import os
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward
import backward


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, shape=[None, forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, shape=[None, forward.OUTPUT_NODE])
        # 测试准确率阶段不需要正则化
        y = forward.forward(x, None)

        # 读取模型文件中滑动平均参数的影子值
        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        saver = tf.train.Saver(ema.variables_to_restore())

        # 计算准确率
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 根据文件名提取global_step值
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # 达到训练最大值，跳出循环
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print('Iter {}, test accuracy is {}'.format(global_step, accuracy_score))
                else:
                    print('No checkpoint file found!')
            time.sleep(5)


def main():
    mnist = input_data.read_data_sets('./data/', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    main()
