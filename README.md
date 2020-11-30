# Gazebo/ROS Gym

This repository implements various Reinforcement Learning environments and provides a framework
for easily implementing new ones.

Three different ROS packages are included:

* gazebo_gym is the main package, and provides the means to easily implement RL ROS environments,
and the various demos.
* gazebo_gym_env_plugin implements a gazebo plugin for controlling gazebo simulations according
 to the needs of RL methods. In particular it allows to step the simulation of precise durations,
 to obtain renderings from simulated cameras even while the simulation is stopped and reduces
 communication overhead making simulation MUCH faster.
* gazebo_gym_utils provides a node that can receive and execute moveit commands via ROS messaging.
 This is needed as an intermediate node for controlling moveit from python3 code.


## The gazebo_gym package

In the gazebo_gym/src/gazebo_gym folder you can find python3 code for implementing ROS-based
RL environments.
The implementation of the environments has been split in two logically separated parts.

* The environment classes, derived from envs/BaseEnv, define the environment in its high-level
 characteristics. Such as the state/observation definition, the reward function definition and
 the initial state.
* The environment controllers, derived from EnvironmentController define the low-level, interfacing
 with the simulators or with the real world.

This is not possible for all cases, but environment controllers are meant to be interchangeable. This can
allow to implement an environment at a high abstraction level and run it without changes on different
simulators, or even in the real world. An example of this is the HopperEnv environment, which can run in
both Gazebo and PyBullet.

Different environment controllers have been implemented for different simulators and for the real world.
All of the environments are derived form EnvironmentController

* GazeboController and GazeboControllerNoPlugin provide the means to control a Gazebo simulation
* PyBulletController allows to control a PyBullet simulation
* RosEnvController uses ROS to control and observe the environment, this is meant to be usable both for
simulations and for the real world (but it will be less efficient and "precise" than GazeboController)
* EffortRosControlController is built on top of RosEnvController and uses ros_control effort controllers to
command joint efforts


## Setup
This repository requires ROS Melodic and python3.6 or later.
To set up the project you first of all need to clone the repository in your catkin
workspace src folder.
You will also need to install some python3 modules, preferably in a python virtual
environment. You can use the requirements.txt file in the gazebo_gym folder:

```
pip3 install -r src/gazebo_gym/gazebo_gym/requirements_sb.txt
```

You can then proceed to build the workspace with `catkin build`.

## Usage - Cartpole

The environment can be tested using a python script that executes a hard-coded policy
through the python gym interface:

```
rosrun gazebo_gym test_cartpole_env.py
```

By default the script does not use the simulated camera, it is possible to enable
it with:

```
rosrun gazebo_gym test_cartpole_env.py --render
```

The rendered frames can be saved to file by specifying the --saveframes option.
The simulation step length can be changed using the --steplength option (the default is 0.05s)



It is also possible to train a basic DQN policy by using the following:

```
rosrun gazebo_gym solve_dqn_stable_baselines.py
```

To see the simulation you can launch a gazebo client using the following:

```
GAZEBO_MASTER_URI=http://127.0.0.1:11414 ROS_MASTER_URI=127.0.0.1:11350 gzclient
```

## Usage - Hopper

You can execute a pre-trained policy with:
```
rosrun gazebo_gym solve_hopper.py --load src/gazebo_gym/gazebo_gym/trained_models/td3_hopper_20200605-175927s140000gazebo-urdf-gazeboGym.zip
```

You can train a new TD3 policy with

```
rosrun gazebo_gym solve_hopper.py
```

You can specify the number of timesteps to train for with the --iterations option.

You can also run the hopper environment directly in pybullet specifynig the --pybullet option, in this mode you can also use an mjcf model (adapted from the OpenAI gym one) instead of the urdf one, using the --mjcf option. Trained models are also provided for these two modes.


## Moveit Panda Reaching Environment

The PandaMoveitReachingEnv.py file implements and environment in which a Panda arm moves using Moveit and has to reach a
specific pose with its end-effector.
Right now it has only been tested within a Gazebo simulation but in future it is supposed fto work also in the real.
It may also be extended to use a camera.

To have the environment working you will need additional ROS packages in your workspace.
First of all the panda package, which provides the model and controller definitions for the Gazebo simulation.
You can clone it as:

```
git clone -b crzz-dev https://gitlab.idiap.ch/learn-real/panda.git
```

You will also need the panda moveit configuration. You can get it by cloning:

```
git clone https://github.com/erdalpekel/simple_panda_moveit_config.git
```

At this point, after compiling with catkin, you can run the environment.

First, launch:
```
roslaunch panda panda_generic_control.launch simulated:=true
```

Then, in a separate terminal:
```
roslaunch simple_panda_moveit_config move_group.launch
```
At this point if you want you can start rviz and control the robot with moveit with rviz -d src/panda/config/moveit.rviz
If the robot is in collision with itself, you can move it to a safe pose with (kill it with ctrl-c after the robot has moved):
```
rosrun panda move_to_start_pose_raw.sh
```

Then, in another separate terminal:
```
rosrun gazebo_gym_utils move_helper
```

Finally, you can start training the RL policy with:
```
rosrun gazebo_gym solve_pandaReaching.py
```

The hyperparameters have not been tuned carefully, so the training may not achieve optimal results.
Also, you can speed up the simulation in Gazebo, clicking in the left panel 'Physics' and
entering -1 in "real time update"





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
