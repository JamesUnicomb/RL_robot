#!/usr/bin/env python

import os, time

import rospy, rospkg

import numpy as np
from math import *
import cv2

from cv_bridge import CvBridge, CvBridgeError

import theano
import theano.tensor as T

import lasagne
from lasagne.updates import nesterov_momentum
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, DenseLayer, FlattenLayer, InputLayer, get_output
from lasagne.nonlinearities import rectify, softmax, tanh

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import scipy.io as io

class PathFollow:
    def __init__(self,
                 load_model    = False,
                 record_data   = False,
                 is_training   = False,
                 max_lin_vel   = 0.5,
                 max_ang_vel   = 0.8,
                 min_lin_vel   = 0.15):

        self.pkg_path = rospkg.RosPack().get_path('rl_path_follow')

        def floatX(X):
            return np.asarray(X, dtype=theano.config.floatX)

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
            def cnn_model(X):
                l_in        = InputLayer(input_var = X,
                                         shape     = (None, 1, 60, 80))

                conv_1      = Conv2DLayer(l_in,
                                          num_filters  = 8,
                                          filter_size  = (5,5),
                                          stride       = (1,1),
                                          nonlinearity = rectify)

                pool_1      = MaxPool2DLayer(conv_1,
                                             pool_size = (3,3),
                                             stride    = (2,2))

                conv_2      = Conv2DLayer(pool_1,
                                          num_filters  = 16,
                                          filter_size  = (4,4),
                                          stride       = (1,1),
                                          nonlinearity = rectify)

                pool_2      = MaxPool2DLayer(conv_2,
                                             pool_size = (3,3),
                                             stride    = (2,2))

                conv_3      = Conv2DLayer(pool_2,
                                          num_filters  = 32,
                                          filter_size  = (3,3),
                                          stride       = (1,1),
                                          nonlinearity = rectify)

                pool_3      = MaxPool2DLayer(conv_3,
                                             pool_size = (3,3),
                                             stride    = (2,2))

                flatten     = FlattenLayer(pool_3)

                hidden_1    = DenseLayer(flatten,
                                         num_units    = 200,
                                         nonlinearity = tanh)

                hidden_2    = DenseLayer(hidden_1,
                                         num_units    = 100,
                                         nonlinearity = tanh)

                output      = DenseLayer(hidden_2,
                                         num_units    = 1,
                                         nonlinearity = tanh)

                return output

            self.X = T.tensor4()
            self.Y = T.fmatrix()

            self.network_output = cnn_model(self.X)

            if load_model:
                with np.load(self.pkg_path + '/model/model.npz') as f:
                    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
                lasagne.layers.set_all_param_values(self.network_output, param_values)

            self.output    = get_output(self.network_output)

            self.prediction = theano.function(inputs               = [self.X],
                                              outputs              = self.output,
                                              allow_input_downcast = True)
             


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

        self.input_image = np.array(self.input_image).reshape(1,1,60,80) * (1.0 / 255)              # reshape and preprocess input image.

        self.feed_dict = {self.X: self.input_image}                                                 # pass the image to the neural network.


    def cmd_cb(self, cmd):
        self.nu = cmd.linear.x                                                                      # recorded linear velocity.
        self.om = cmd.angular.z                                                                     # recorded angular velocity.
        
        if self.nu > self.min_lin_vel:
            self.y = (self.om / self.nu)                                                            # path curvature. to be predicted by network. (OUTPUT)


    def set_velocity(self,
                     linear_velocity = 0.0):

        output_twist = Twist()

        network_output = 0.0

        if self.input_image.shape == (1,1,60,80):
            input_X = (self.input_image).reshape(1,1,60,80)
            network_output = self.prediction(input_X)

        output_twist.linear.x = linear_velocity
        output_twist.angular.z = network_output

        self.vel_pub.publish(output_twist)     


    def return_data(self):
        return (self.nu, self.om, self.y), self.input_image


    def train_model(self,
                    lr         = 0.0001,
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

        trX = np.array(trX).reshape(-1,1,60,80)
        trY = np.array(trY)[:,1].reshape(-1,1)

        print trX.shape
        print trY.shape

        l1 = np.abs(self.output - self.Y)
        l2 = np.square(self.output - self.Y)

        loss = alpha * l1 + (1.0 - alpha) * l2

        loss = loss.mean()

        params = lasagne.layers.get_all_params(self.network_output, 
                                               trainable=True)

        updates = nesterov_momentum(loss, 
                                    params, 
                                    learning_rate = lr,
                                    momentum      = 0.9)

        train = theano.function(inputs=[self.X, self.Y], 
                                outputs=loss, 
                                updates=updates, 
                                allow_input_downcast=True)

        for epoch in range(n_epochs):
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
                train(trX[start:end], trY[start:end])
            tl1 = np.mean(np.abs(self.prediction(trX) - trY))
            tl2 = np.mean(np.square(self.prediction(trX) - trY))
            train_loss = alpha * tl1 + (1.0 - alpha) * tl2
            print('epoch: %d, loss: %f' % (epoch, train_loss))
        
        np.savez(self.pkg_path + '/model/model.npz', *lasagne.layers.get_all_param_values(self.network_output))
