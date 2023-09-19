# Learn-Real Gym

This repository implements various Reinforcement Learning environments and provides a framework
for easily implementing new ones.

Three different ROS packages are included:

* lr_gym is the main package, and provides the means to easily implement RL ROS environments,
and the various demos.
* gazebo_gym_env_plugin implements a gazebo plugin for controlling gazebo simulations according
 to the needs of RL methods. In particular it allows to step the simulation of precise durations,
 to obtain renderings from simulated cameras even while the simulation is stopped and reduces
 communication overhead making simulation MUCH faster.
* lr_gym_utils provides various utilities for helping development. Important ones are:
    - A node that can receive and execute moveit commands via ROS messaging. This is needed as an intermediate
      node for controlling moveit from python3 code.
    - A ros controller that allows to perform effort control on top of a gravity-compensation controller
    - A ros controller that publishes link state information (e.g. cartesian poses of links)



## The lr_gym package

In the lr_gym/src/lr_gym folder you can find python3 code for implementing ROS-based
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
All of the environments controllers are derived form EnvironmentController

* **GazeboController** and **GazeboControllerNoPlugin** provide the means to control a Gazebo simulation
* **PyBulletController** allows to control a PyBullet simulation
* **RosEnvController** uses ROS to control and observe the environment, this is meant to be usable both for
simulations and for the real world (but it will be less efficient and "precise" than GazeboController)
* **EffortRosControlController** is built on top of RosEnvController and uses ros_control effort controllers to
command joint efforts
* **MoveitRosController** is built on top of RosEnvController and uses MoveIt to control a robot in cartesian space
* **MoveitGazeboController** is build on top of MoveitRosController and GazeboController, it integrates the moveit-based
 control with useful methods only available in simulation

There are various environments already implemented, you can find them in the `lr_gym/src/lr_gyn/envs` folder.


## Setup - stable_baselines 3
These instructions assume you are on ROS noetic and Ubuntu 20.04.

To set up the project you first of all need to clone the following repositories in your
catkin workspace src folder, be careful about using the correct branch and correct folder name:

 * This current repository (if you didn't do it already):
   ```
   git clone --depth 1 --branch frontiers_2210 https://github.com/c-rizz/lr_gym
   ```
 * For using the Panda arm you will need the `lr_panda` repository:
   ```
   git clone --depth 1 --branch frontiers_2210 https://github.com/c-rizz/lr_panda
   ```
 * For using the Panda arm with moveit you will need `lr_panda_moveit_config`:
   ```
   git clone --depth 1 --branch frontiers_2210 https://github.com/c-rizz/lr_panda_moveit_config
   ```
 * For using the realsense camera you will need `lr_realsense`:
   ```
   git clone --depth 1 --branch frontiers_2210 https://github.com/c-rizz/lr_realsense
   ```

You will also need to install some python3 modules, preferably in a python virtual
environment. You can use the build_virtualenv.sh helper script in the lr_gym folder.


You can create the virtual python environment with the following helper
script:

```
src/lr_gym/lr_gym/build_virtualenv.sh sb3
```

At this point you can enter the virtual environment with:

```
. ./virtualenv/lr_gym_sb/bin/activate
```


You can then install all the ros dependencies with rosdep (you may
 need to install python3-rosdep from apt, *DO NOT* install python3-rosdep2, it would break your ROS installation):

```
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```


For building the workspace I use catkin build. To have catkin build
you can install catkin-tools, at the moment it can be done with:

```
sudo apt install python3-catkin-tools python3-osrf-pycommon
```

You can now build with:

```
catkin build
```

To use in headless setups you may want to use pyvirtualdisplay, for it to work you
will need to install the following system dependencies (however you will not have GPU acceleration for rendering if you use it):

```
sudo apt install xvfb xserver-xephyr tigervnc-standalone-server xfonts-base
```

### Installation issues

#### CUDA error: no kernel image is available for execution on device
You may get this error if you are using a relatively new card (or maybe a very old one). I had this problem on an RTX3060 and on an RTX A6000.
This is due to your pytorch installation not being compiled for the architecture of your gpu (e.g for sm_86).
You can get a build that includes your device from pytorch.org, but you need to select the correct torch/torchvision version and the correct CUDA version. You can get torch and torchvision from requirements_sb3.txt in the lr_gym folder. I believe you should use the CUDA version that appears in nvidia-smi, but that exact one may not be available on pytorch.org, so choose a similar one. I for example had CUDA 11.5, but installed pytorch for CUDA 11.3, and everything worked. You can find the list of the builds at https://download.pytorch.org/whl/torch_stable.html
You can then install with the following (replace the versions according to your needs):

```
pip install --pre torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

### Basic Training

It is also possible to train a basic DQN policy by using the following:

```
rosrun lr_gym solve_cartpole_sb3.py --controller GazeboController
```

To see the simulation you can launch a gazebo client using the following:

```
roslaunch lr_gym gazebo_client.launch id:=0
```

It should reach the maximum episode length of 500 steps in about 400 episodes.


## Example - Hopper

You can train a new TD3 policy with

```
rosrun lr_gym solve_hopper_sb3.py
```

You can specify the number of timesteps to train for with the --iterations option.

You can also run the hopper environment in pybullet specifying the --pybullet option, in this mode you can also use an mjcf model (adapted from the OpenAI gym one) instead of the urdf one, using the --mjcf option.

As always the Gazebo simulation can be visualized using 'roslaunch lr_gym gazebo_client.launch'

## Examples - Panda Arm

A series of different environments based on the Franka-Emika Panda arm are available.
 * **PandaEffortStayUp** features an effort-controller Panda arm, the objective is to keep the panda arm upright
 * **PandaEffortKeepPose** again features an effort-controller Panda arm, the objective is to keep the panda arm
   end effector in a specific pose that does not vary between episodes
 * **PandaEffortKeepVarPose** again features an effort-controller Panda arm, the objective is to keep the panda arm
   end effector in a specific pose that changes between each episode (this environment has not been solved)
 * **PandaMoveitReachingEnv** features a Panda arm controlled in cartesian space with moveit. The objective is to
   keep the panda arm end effector in a specific pose that does not vary between episodes. **This environment work both in simulation and in the real.**
 * **PandaMoveitVarReachingEnv** features a Panda arm controlled in cartesian space with moveit. The objective is to
   keep the panda arm end effector in a specific pose that changes between each episode. **This environment work both in simulation and in the real.**
 * **PandaMoveitPick** features a Panda arm controlled in cartesian space via Moveit, it is also possible to control the Franka Hand gripper. The goal is to pick an object placed on the ground in front of tthe robot and lift it up. **This environment work both in simulation and in the real.**

Some scripts that solve or test each of these evironments in different ways are available in the examples folder. 

## Plotting

Executing and environment via the GymEnvWrapper allows to log information about the environment execution.
You can create plots of the learnig curves from these log suing the `plotGymEnvLogs.py` script. For example:

```
rosrun lr_gym plotGymEnvLogs.py --csvfile solve_cartpole_env/20210310-120235/GymEnvWrapper_log.csv
```

Check the `rosrun lr_gym plotGymEnvLogs.py --help` for more info

## Gazebo Plugin
To correctly implement the OpenAI gym environment interface, it is necessary to execute
simulation steps of precise length and to obtain renderings synchronized with these
simulation steps. This is needed to accurately implement a Markov Decision Process, which
is the basic abstraction of the environment used in reinforcement learning. While this requirement
may be lifted in general it is necessary for reproducibility and useful for experimentation.


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
the theoretical abstraction of an MDP.

