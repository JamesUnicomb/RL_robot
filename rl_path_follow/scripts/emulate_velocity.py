#!/usr/bin/env python

import rospy
from rl_path_follow import PathFollow

import tensorflow as tf

def main():
    rospy.init_node('path_follow_test')

    pf = PathFollow(load_model = True, record_data = False)

    rate = rospy.Rate(10)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(pf.pkg_path + '/model/model-2000.meta')
        saver.restore(sess, tf.train.latest_checkpoint(pf.pkg_path + '/model/'))

        graph = tf.get_default_graph()

        pf.sess = sess

        pf.X      = graph.get_tensor_by_name('image_input:0')
        pf.Y      = graph.get_tensor_by_name('network_output:0')

        pf.output = graph.get_tensor_by_name('output:0')

        while not rospy.is_shutdown():
            pf.set_velocity(pf.nu)
            rate.sleep()

if __name__=='__main__':
    main()
