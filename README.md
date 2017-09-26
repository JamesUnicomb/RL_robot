# rl_robot
RL_Robot Base Code.

### rl_robot
This puts the drivers of the robot into several launch files.

```
roslaunch rl_robot robot_minimal.launch
```
launches the camera and create drivers.

```
roslaunch rl_robot robot_laser.launch 
```
launches the rplidar, imu and create drivers. This will also run the ekf between the imu and the odometry.

```
roslaunch rl_robot robot_rs.launch
```
launches the realsense, imu and create drivers.

## Robot Mapping
### rl_mapping
The video below shows google cartographer mapping a floor.
```
roslaunch rl_mapping carto_mapping.launch
```

[![Watch the video](https://github.com/JamesUnicomb/rl_robot/blob/master/video_clipping_mapping.png)](https://www.youtube.com/watch?v=EqAxq1zT4jE)
Video Link: https://www.youtube.com/watch?v=EqAxq1zT4jE

With a pre-recorded rosbag of a robot driving around an environment, mapping can be done with this data by the commands in two seperate terminals:
```
roslaunch rl_mapping carto_mapping.launch map_online:=false
rosbag play /path/to/bagfile/file.bag
```

Alternative mapping packages include, gmapping and hector_mapping:
```
roslaunch rl_mapping gmapping.launch
```
```
roslaunch rl_mapping hector_mapping.launch
```


## Robot Navigation
### rl_navigation
#### Navigation in a static home environment
```
roslaunch rl_navigation flat.launch
```

[![Watch the video](https://github.com/JamesUnicomb/rl_robot/blob/master/video_clipping_navigation.png)](https://www.youtube.com/watch?v=eP6TwM0xmTs)
Video Link: https://www.youtube.com/watch?v=eP6TwM0xmTs

#### Navigation in a static maze
```
roslaunch rl_navigation maze.launch
```


## Path Following with Neural Networks
This package uses tensorflow or theano to use recorded data to train a neural network to estimate the turning radius of a robot.

The following video shows the training data set along with the trained neural network output.

[![Watch the video](https://github.com/JamesUnicomb/rl_robot/blob/master/nn_control1.png =100x100)](https://www.youtube.com/watch?v=hoHlfk9FW_M)
Video Link: https://www.youtube.com/watch?v=hoHlfk9FW_M

Once the training has been completed we test how well the neural network predicts the turning radius of someone driving the robot along an unseen track.

Its important to note that the network will not match the human control perfectly as the human control will be jolty. 

[![Watch the video](https://github.com/JamesUnicomb/rl_robot/blob/master/nn_control2.png)](https://www.youtube.com/watch?v=J_xf4Yv39JY)
Video Link: https://www.youtube.com/watch?v=J_xf4Yv39JY

Finally, the video below shows the robot driving along a track autonomously. TBA
