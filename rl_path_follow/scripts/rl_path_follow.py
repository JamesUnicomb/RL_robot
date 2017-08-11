#!/usr/bin/env python

import os, time

import rospy, rospkg

import numpy as np
from math import *
import cv2

from cv_bridge import CvBridge, CvBridgeError

import tensorflow as tf

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import scipy.io as io

class PathFollow:
    def __init__(self,
                 load_model    = True,
                 record_data   = True,
                 is_training   = False,
                 padding       = 'SAME',
                 n_filter_1    = 8,
                 filter_size_1 = 12,
                 pool_size_1   = 2,
                 n_filter_2    = 16,
                 filter_size_2 = 8,
                 pool_size_2   = 2,
                 n_filter_3    = 32,
                 filter_size_3 = 4,
                 pool_size_3   = 2,
                 n_hidden_1    = 128,
                 n_hidden_2    = 64,
                 max_lin_vel   = 0.5,
                 max_ang_vel   = 0.8,
                 min_lin_vel   = 0.15):

        self.pkg_path = rospkg.RosPack().get_path('rl_path_follow')

        self.nu = 0.0                                                                               # linear (forward) velocity.
        self.om = 0.0                                                                               # angular (turn) velocity.
        self.y  = 0.0                                                                               # proportional to inverse curvature.

        self.input_image = np.zeros((1,60,80,1))                                                    # initial image (zeros).

        self.max_lin_vel = max_lin_vel                                                              # maximum and minimum velocites used for curvature and preprocessing and output.
        self.min_lin_vel = min_lin_vel
        self.max_ang_vel = max_ang_vel                                                              # taken from the teleop script.

        self.bridge = CvBridge()

        rospy.Subscriber('camera/rgb/image_rect_color', Image, self.rgb_cb)                         # subscibe to camera image as input. feeds into neural network. (INPUT)

        rospy.Subscriber('cmd_vel', Twist, self.cmd_cb)                                             # subscribe to command velocity. used for network to learn.

        self.transformed_image_pub = rospy.Publisher('network/transformed_ground_image', 
                                                                           Image, queue_size=5)     # publisher to show transformed (birds eye) ground image.

        self.display_image_pub = rospy.Publisher('network/display_ground_image',
                                                                           Image, queue_size=5)     # publisher to show image with used ground space.

        self.vel_pub = rospy.Publisher('network/cmd_vel', Twist, queue_size=5)                      # network output command.

        pts = [[[480,360],[640,480],[0,480],[160,360]],
               [[640,0],[640,480],[0,480],[0,0]]]                                                   # transoform for points onto ground plane

        self.d_pts = np.array(pts[0])

        self.M = cv2.getPerspectiveTransform(np.float32(pts[0]), np.float32(pts[1]))

        if not record_data:
            self.sess = tf.Session()

            if load_model:
                saver = tf.train.import_meta_graph(self.pkg_path + '/model/model-2000.meta')
                saver.restore(self.sess, tf.train.latest_checkpoint(self.pkg_path + '/model/'))

                graph = tf.get_default_graph()

                op = self.sess.graph.get_operations()
                
                troubleshoot = True
                if troubleshoot:
                    for m in op:
                        print m.values()

                self.X      = graph.get_tensor_by_name('image_input:0')
                self.Y      = graph.get_tensor_by_name('network_output:0')

                self.output = graph.get_tensor_by_name('output:0')


            else:
                self.X = tf.placeholder(tf.float32, [None, 60, 80, 1], name='image_input')
                self.Y = tf.placeholder(tf.float32, [None, 1], name='network_output')

                self.w_c1 = tf.Variable(tf.random_normal([filter_size_1,
                                                          filter_size_1, 
                                                          1, 
                                                          n_filter_1]),
                                                          name='c1_weights')

                self.w_c2 = tf.Variable(tf.random_normal([filter_size_2, 
                                                          filter_size_2, 
                                                          n_filter_1, 
                                                          n_filter_2]),
                                                          name='c2_weights')

                self.w_c3 = tf.Variable(tf.random_normal([filter_size_3, 
                                                          filter_size_3, 
                                                          n_filter_2, 
                                                          n_filter_3]),
                                                          name='c3_weights')
                
                self.c1     = tf.nn.conv2d(self.X, 
                                           self.w_c1, 
                                           strides=[1,6,6,1], 
                                           padding=padding)

                self.p1     = tf.nn.max_pool(self.c1, 
                                             ksize=[1,2,2,1], 
                                             strides=[1,2,2,1], 
                                             padding=padding)

                self.c2     = tf.nn.conv2d(self.p1, 
                                           self.w_c2, 
                                           strides=[1,4,4,1], 
                                           padding=padding)

                self.p2     = tf.nn.max_pool(self.c2, 
                                             ksize=[1,2,2,1], 
                                             strides=[1,2,2,1], 
                                             padding=padding)

                self.c3     = tf.nn.conv2d(self.p2, 
                                           self.w_c3, 
                                           strides=[1,2,2,1], 
                                           padding=padding)

                self.p3     = tf.nn.max_pool(self.c3, 
                                             ksize=[1,2,2,1], 
                                             strides=[1,2,2,1], 
                                             padding=padding)

                self.f1     = tf.contrib.layers.flatten(self.p3)

                self.h1     = tf.contrib.layers.fully_connected(self.f1, 
                                                                n_hidden_1, 
                                                                activation_fn=tf.tanh)
     
                self.h2     = tf.contrib.layers.fully_connected(self.h1, 
                                                                n_hidden_2, 
                                                                activation_fn=tf.tanh)

                self.d1 = tf.layers.dense(self.h2, 1, activation=None)

                self.output = tf.tanh(self.d1, name='output')

            self.feed_dict = {self.X: self.input_image}

            if not is_training:
                self.init = tf.global_variables_initializer()
                self.sess.run(self.init)


    def rgb_cb(self, image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image)
        except CvBridgeError as e:
            print(e)

        self.input_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)                               # turn image to grayscale.
        self.input_image = cv2.warpPerspective(self.input_image,self.M,(640,480))                   # apply transformation to get approximate birds eye view.                      
        self.input_image = cv2.resize(self.input_image, (0,0), fx=0.125, fy=0.125)                  # compress image to a lower size.


        cv2.polylines(cv_image, [self.d_pts], 1, (255, 0, 0), thickness=5)                          # add outline of selected section for birds eye view.

        try:
            self.transformed_image_pub.publish(self.bridge.cv2_to_imgmsg(self.input_image))         # publish image for display and trouble shooting.
        except CvBridgeError as e:
            print(e)

        try:
            self.display_image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image))                     # publish image for display and trouble shooting.
        except CvBridgeError as e:
            print(e)

        self.input_image = np.array(self.input_image).reshape(1,60,80,1) * (1.0 / 255)              # reshape and preprocess input image.

        self.feed_dict = {self.X: self.input_image}                                                 # pass the image to the neural network.


    def cmd_cb(self, cmd):
        self.nu = cmd.linear.x                                                                      # recorded linear velocity.
        self.om = cmd.angular.z                                                                     # recorded angular velocity.
        
        if self.nu > self.min_lin_vel:
            self.y = (self.om / self.nu)                                                            # path curvature. to be predicted by network. (OUTPUT)


    def set_velocity(self,
                     linear_velocity = 0.0):

        output_twist = Twist()

        network_output = self.sess.run(self.output, feed_dict=self.feed_dict)

        output_twist.linear.x = linear_velocity
        output_twist.angular.z = network_output

        self.vel_pub.publish(output_twist)     


    def return_data(self):
        return (self.nu, self.om, self.y), self.input_image


    def train_model(self,
                    lr         = 0.00001,
                    n_epochs   = 1000,
                    batch_size = 128,
                    alpha      = 0.5):

        file_list = os.listdir(self.pkg_path + '/data/training_data')

        trX = []
        trY = []

        for f in file_list:
            data = io.loadmat(self.pkg_path + '/data/training_data/' + f + '/training_data.mat')
            trX = data['images']
            trY = data['velocities']

        trX = np.array(trX)
        trY = np.array(trY)[:,1].reshape(-1,1)

        print trX.shape
        print trY.shape

        l2 = tf.reduce_mean(tf.square(self.output - self.Y))
        l1 = tf.reduce_mean(tf.abs(self.output - self.Y))

        loss = alpha * l1 + (1.0 - alpha) * l2

        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        saver = tf.train.Saver()

        for epoch in range(n_epochs):
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
                self.sess.run(optimizer, feed_dict={self.X: trX[start:end], self.Y: trY[start:end]})
            train_loss = self.sess.run(loss, feed_dict={self.X: trX, self.Y: trY})
            print('epoch: %d, loss: %f' % (epoch, train_loss))
                
        saver.save(self.sess, self.pkg_path + '/model/model', global_step=n_epochs)
