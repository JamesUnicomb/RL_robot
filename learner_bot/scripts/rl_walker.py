#!/usr/bin/env python

import rospy, rospkg

from create_env import createAPI
from learner_bot import RLData

rospy.init_node('policy_learning')

robot_data = RLData()
create = createAPI()

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

from std_msgs.msg import Float64

reset_pub = rospy.Publisher('create/reset', Float64, queue_size=2)

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('learner_bot')

n_inputs = 80
n_hidden1 = 64
n_hidden2 = 32
n_outputs = 3

learning_rate = 0.1

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

hidden1 = fully_connected(X, n_hidden1, activation_fn=tf.nn.relu, weights_initializer=initializer)
hidden2 = fully_connected(hidden1, n_hidden2, activation_fn=tf.nn.relu, weights_initializer=initializer)
logits = fully_connected(hidden2, n_outputs, activation_fn=tf.nn.relu)
probs = tf.nn.softmax(logits)
action = tf.multinomial(logits, num_samples=1)
y = tf.to_float(probs)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

def vel_from_int(input):
    lin_vel = 0.0
    ang_vel = 0.0
    if input==0:
        lin_vel = 0.2
    elif input==1:
        ang_vel = -0.8
    elif input==2:
        ang_vel = 0.8
    return lin_vel, ang_vel

n_games_per_update = 10
n_max_steps = 60
n_iterations = 250
save_iterations = 3
discount_rate = 0.9

def main():
    rate = rospy.Rate(10)

    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            print 'iteration: %d' % (iteration)
            all_rewards = []
            all_gradients = []
            for game in range(n_games_per_update):
                current_rewards = []
                current_gradients = []
                #create.random_walkler(10.0)
                step = 0
                robot_feedback = False
                obs = robot_data.laser_array
                while (step<n_max_steps)and(not robot_feedback):
                    action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs.reshape(1, n_inputs)})
                    action_int = np.argmax(action_val)

                    lin_vel, ang_vel = vel_from_int(action_int)

                    create.set_velocity(lin_vel, ang_vel)

                    robot_feedback = (create.left_bumper)or(create.right_bumper)
                    obs = robot_data.laser_array

                    reward = 2 * (action_int==0) - 50 * robot_feedback

                    current_rewards.append(reward)
                    current_gradients.append(gradients_val)
                    
                    step += 1

                    rate.sleep()

                all_rewards.append(current_rewards)
                all_gradients.append(current_gradients)

            all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate=discount_rate)
            feed_dict = {}
            for var_index, gradient_placeholder in enumerate(gradient_placeholders):
                mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                          for game_index, rewards in enumerate(all_rewards)
                                              for step, reward in enumerate(rewards)], axis=0)
                feed_dict[gradient_placeholder] = mean_gradients
            sess.run(training_op, feed_dict=feed_dict)
            if iteration % save_iterations == 0:
                saver.save(sess, "./my_policy_net_pg.ckpt")

        


if __name__=='__main__':
    main()
