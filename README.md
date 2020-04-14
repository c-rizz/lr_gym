# Gazebo Gym Environment

This repository provides a demo setup which uses Gazebo and ROS to simulate a cart-pole task that
implements the python OpenAI gym environment interface.

## Setup
This repository requires ROS Melodic and python3.6 or later.
To set up the project you first of all need to clone the repository in your catkin
workspace src folder.
You will also need to install some python3 modules, preferably in a python virtual
environment. You can use the requirements.txt file in the demo_gazebo folder:

```
pip3 install -r demo_gazebo/requirements.txt
```

You can then proceed to build the workspace with catkin_make.

##Usage
The environment can be launched with:

```
roslaunch gazebo_demo cartpole_gazebo_sim.launch
```

It is possible to specify the argument gui:=false to run the simulation without the
Gazebo graphical interface


The environment can be tested using a python script that executes a hard-coded policy
through the python gym interface:

```
rosrun demo_gazebo test_cartpole_env.py
```

By default the script does not use the simulated camera, it is possible to enable
it with:

```
rosrun demo_gazebo test_cartpole_env.py --render
```

The rendered frames can be saved to file by specifying the --saveframes option.
The simulation step length can be changed using the --steplength option (the default is 0.05s)



It is also possible to train a basic DQN policy by using the following:

```
rosrun demo_gazebo solve_dqn_stable_baselines.py
```



## Gazebo Plugin
To correctly implement OpenAI gym environment interface, it is necessary to execute
simulation steps of precise length and to obtain renderings synchronized with these
simulation steps. This is needed to accurately implement a Markov Decision Process, which
is the basic abstraction of the environment used in reinforcement learning.


This has been implemented by creating a C++ Gazebo plugin. The plugin is implemented as
a ROS package in the gazebo_gym_env_plugin folder.
The plugin provides two new services:

- /gazebo/gym_env_interface/step allows to step the simulation of a precise amount
 of iterations or for a specific duration
- /gazebo/gym_env_interface/render allows to obtain a rendering from the simulated
 cameras within the simulation. This can be called even while the simulation is paused.
 This provides a way to obtain renderings that precisely correspond to the simulation steps.


Controlling the simulation using these services is not realistic (the real world
can not be stopped like the step function does), but they allow to closely approximate
the theoretical abstraction used in reinforcement learning.


## Performance
The repository was tested on two laptops:

1. An Alienware laptop equipped with an Intel i7-6820HK CPU and an Nvidia GeForce GTX 980M GPU.
2. An HP Omen laptop equipped with an Intel i7-8750H CPU and an NVIDIA GeForce GTX 1050 Ti Mobile GPU.

The performance on the two computers is as follows:

| Laptop |       No Rendering        | With Rendering          |
|--------|---------------------------|-------------------------|
|   1    |  45fps (2.25x real time)  | 23fps (1.17x real time) |
|   2    |  63fps (3.14x real time)  | 27fps (1.35x real time) |

The simulation step was kept at 0.05s for all the tests.

The Alienware laptop was also used for determining were most of the computation time
is spent. A simulation speed of 23fps means each step requires about 43ms. The rendering
requires about 10ms, the physics simulation takes about 15ms. The remaining 18ms are
consumed by the ROS messaging overhead: about 10ms for the rendered image transfer
and the rest for gathering the joint states and for sending the joint effort commands.
So, in the case of the Alienware laptop, 25% of the time is spent in the rendering,
35% is spent in the physics simulation, the remaining 40% is spent in joint state
reading, joint effort control, and messaging overhead.
