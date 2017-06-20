# rl_robot
RL_Robot Base Code.


## Robot Mapping
### rl_mapping
The video below shows google cartographer mapping a floor.

[![Watch the video](https://github.com/JamesUnicomb/rl_robot/blob/master/video_clipping_mapping.png)](https://www.youtube.com/watch?v=EqAxq1zT4jE)
Video Link: https://www.youtube.com/watch?v=EqAxq1zT4jE


## Robot Navigation
### rl_navigation
The video below shows the navigation stack working in a static home environment.

[![Watch the video](https://github.com/JamesUnicomb/rl_robot/blob/master/video_clipping_navigation.png)](https://www.youtube.com/watch?v=eP6TwM0xmTs)
Video Link: https://www.youtube.com/watch?v=eP6TwM0xmTs


## Path Following with Neural Networks
This package uses tensorflow or theano to use recorded data to train a neural network to estimate the turning radius of a robot.

The following video shows the training data sets.

Once the training has been completed we test how well the neural network predicts the turning radius of someone driving the robot along an unseen track.

Finally, the video below shows the robot driving along a track autonomously.
