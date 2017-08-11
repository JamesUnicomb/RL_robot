#!/usr/bin/env python

import os, time

import rospy
from rl_path_follow import PathFollow

import tensorflow as tf

import scipy.io as io
import numpy as np

def main():
    rospy.init_node('path_follow_test')

    pf = PathFollow(load_model = True, record_data = False)

    data = io.loadmat(pf.pkg_path + '/data/training_data/' + os.listdir(pf.pkg_path + '/data/training_data')[0] + '/training_data.mat')
    trX = data['images']
    trY = data['velocities']

    trX = np.array(trX)
    trY = np.array(trY)[:,1].reshape(-1,1)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(pf.pkg_path + '/model/model-2000.meta')
        saver.restore(sess, tf.train.latest_checkpoint(pf.pkg_path + '/model/'))

        graph = tf.get_default_graph()

        pf.X      = graph.get_tensor_by_name('image_input:0')
        pf.Y      = graph.get_tensor_by_name('network_output:0')

        pf.output = graph.get_tensor_by_name('output:0')

        l2 = tf.reduce_mean(tf.square(pf.output - pf.Y))
        l1 = tf.reduce_mean(tf.abs(pf.output - pf.Y))

        alpha = 0.5

        loss = alpha * l1 + (1.0 - alpha) * l2

        train_loss = sess.run(loss, feed_dict={pf.X: trX, pf.Y: trY})

        print train_loss

    

if __name__=='__main__':
    main()
