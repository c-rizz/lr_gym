# Gazebo Gym Environment

This repository provides a demo setup which implements the python OpenAI gym environment
interface using Gazebo and ROS.

Two different environments are included: a Cartpole environment and a Hopper environment.

## Setup
This repository requires ROS Melodic and python3.6 or later.
To set up the project you first of all need to clone the repository in your catkin
workspace src folder.
You will also need to install some python3 modules, preferably in a python virtual
environment. You can use the requirements.txt file in the demo_gazebo folder:

```
pip3 install -r src/gazebo_gym/demo_gazebo/requirements.txt
```

You can then proceed to build the workspace with catkin_make.

## Usage - Cartpole

The cartpole environment can be launched with:

```
roslaunch demo_gazebo cartpole_gazebo_sim.launch
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

## Usage - Hopper

You can launch the hopper Gazebo environment with
```
roslaunch demo_gazebo hopper_gazeno_sim.launch
```

You can execute a pre-trained policy with:

```
rosrun demo_gazebo solve_hopper.py --load src/demo_gazebo/demo_gazebo/trained_models/td3_hopper_20200605-175927s140000gazebo-urdf-gazeboGym.zip
```

You can train a new TD3 policy with

```
rosrun demo_gazebo solve_hopper.py
```

You can specify the number of timesteps to train for with the --iterations option.

You can also run the hopper environment directlyin pybullet specifynig the --pybullet option, in this mode you can also use an mjcf model (adapted from the OpenAI gym one) instead of the urdf one, using the --mjcf option. Trained models are also provided for these two modes.


## Gazebo Plugin
To correctly implement the OpenAI gym environment interface, it is necessary to execute
simulation steps of precise length and to obtain renderings synchronized with these
simulation steps. This is needed to accurately implement a Markov Decision Process, which
is the basic abstraction of the environment used in reinforcement learning.


This has been implemented by creating a C++ Gazebo plugin. The plugin is implemented as
a ROS package in the gazebo_gym_env_plugin folder.
The plugin provides two new services:

- /gazebo/gym_env_interface/step allows to step the simulation of a precise amount
 of iterations or for a specific duration
- /gazebo/gym_env_interface/render allows to obtain a rendering from the simulated
 cameras. This can be called even while the simulation is paused.
 This provides a way to obtain renderings that precisely correspond to the simulation steps.

The step service also allows to obtain observations of joint/link states and renderings. Using this functionality allows to reduce drastically the communication overhead.

Controlling the simulation using these services is not realistic (the real world
can not be stopped like the step function does), but they allow to closely approximate
the theoretical abstraction used in reinforcement learning.


## Performance
The repository was tested on two laptops:

1. An Alienware laptop equipped with an Intel i7-6820HK CPU and an Nvidia GeForce GTX 980M GPU.
2. An HP Omen laptop equipped with an Intel i7-8750H CPU and an NVIDIA GeForce GTX 1050 Ti Mobile GPU.

Here is reported the performance of the test_cartpole_env script.

| Laptop |       No Rendering        | With Rendering            |
|--------|---------------------------|---------------------------|
|   1    |  370fps (15.3x real time) | 41.5fps (2.02x real time) |
|   2    |  531fps (22.1x real time) | 44.3fps (2.17x real time) |

The simulation step was kept at 0.05s for all the tests.
