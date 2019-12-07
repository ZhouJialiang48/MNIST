#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/7 10:31
# @Author  : zhoujl
# @Site    : 
# @File    : evaluation.py
# @Software: PyCharm
import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward
import backward


def predict(data):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, shape=[None, forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, shape=[None, forward.OUTPUT_NODE])
        # 测试准确率阶段不需要正则化
        y = forward.forward(x, None)

        # 读取模型文件中滑动平均参数的影子值
        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        saver = tf.train.Saver(ema.variables_to_restore())

        # 预测数字与真实数字
        true = tf.argmax(y_, 1)
        predicted = tf.argmax(y, 1)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 根据文件名提取global_step值
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                # 达到训练最大值，跳出循环
                predicted_number = sess.run(predicted, feed_dict={x: data['images']})
                true_number = sess.run(true, feed_dict={y_: data['labels']})
                print('Iter {}, predicted number is {}, true number is {}.'.format(global_step, predicted_number[0], true_number[0]))
            else:
                print('No checkpoint file found!')


# 数据加载与预处理
def load_data_handwriting(i):
    """加载手写数字图片

    1. 加载图片，转化为灰度图
    2. 压缩到指定大小(28*28)，转化为矩阵形式，归一化像素值，并过滤噪点
    3. 转换维度，28*28转化为1*784，以符合模型的输入维度

    Args:
        i: ./img/目录下的对应数字图片，当前示例图片文件名称为：2.png

    Returns:
        包含特征和标签的字典类型数据包
    """

    # 以灰度模式加载原始图片
    im_raw = Image.open('img/{number}.png'.format(number=i)).convert('L')

    # 压缩图片，并转化成数据矩阵
    im_small = im_raw.resize((28, 28), Image.ANTIALIAS)
    im = (1 - np.asarray(im_small)) / 255
    im = np.where(im < 0.6, 0, im)

    # 处理成模型可读取的格式
    imgs = im.reshape(1, 784)

    labels = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    ])

    return {
        'images': imgs,
        'labels': labels
    }


def main():
    data = load_data_handwriting(2)
    predict(data)


if __name__ == '__main__':
    main()
