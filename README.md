# Gazebo Gym Environment

This repository provides a demo of Gazebo used to simulate a cart-pole task that
implements the python OpenAI gym environment interface.
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
The step length can be changed using the --steplength option (the default is 0.05s)



It is also possible to train a basic DQN policy by using the following:

```
rosrun demo_gazebo solve_dqn_stable_baselines.py
```



## Gazebo Plugin
To correctly implement OpenAI gym environment interface, it is necessary to execute
simulation steps of precise length and to obtain renderings synchronized with these
simulation steps. This is needed to accurately implement a Markov process, which
is the basic abstraction of the environment used in reinforcement learning.


This has been implemented by creating a Gazebo plugin. The plugin is implemented as
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
The repository was tested on a laptop equipped with an Intel i7-6820HK CPU and a
Nvidia GeForce GTX 980M GPU.

The simulation, with a step length of 0.05s and without rendering runs at about 45fps,
which corresponds to about 2.25x real time.

Enabling the rendering reduces the frame rate to about 23fps, about 1.2x real time.

A simulation speed of 23fps means each step requires about 43ms. The rendering requires
about 10ms, the physics simulation takes about 15ms. The remaining 18ms are consumed by
the ROS messaging overhead: about 10ms for the rendered image transfer and the rest for
gathering the joint states and for sending the joint effort commands.
