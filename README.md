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

* **GazeboController** and GazeboControllerNoPlugin provide the means to control a Gazebo simulation
* **PyBulletController** allows to control a PyBullet simulation
* **RosEnvController** uses ROS to control and observe the environment, this is meant to be usable both for
simulations and for the real world (but it will be less efficient and "precise" than GazeboController)
* **EffortRosControlController** is built on top of RosEnvController and uses ros_control effort controllers to
command joint efforts
* **MoveitRosController** is built on top of RosEnvController and uses MoveIt to control a robot in cartesian space
* **MoveitGazeboController** is build on top of MoveitRosController and GazeboController, it integrates the moveit-based
 control with useful methods only available in simulation

There are various environments already implemented, you can find them in the `lr_gym/src/lr_gyn/envs` folder.


## Setup - stable_baselines 2
These instructions assume you are on ROS noetic and Ubuntu 20.04.

To set up the project you first of all need to clone the following repositories in your
catkin workspace src folder:

 * This current repository (if you didn't do it already):
   ```
   git clone --branch master https://gitlab.idiap.ch/learn-real/lr_gym.git
   ```
 * For using the Panda arm you will need the `lr_panda` repository:
   ```
   git clone --branch crzz-dev https://gitlab.idiap.ch/learn-real/panda.git lr_panda
   ```
 * For using the Panda arm with moveit you will need `lr_panda_moveit_config`:
   ```
   git clone --branch crzz-dev https://gitlab.idiap.ch/learn-real/lr_panda_moveit_config.git lr_panda_moveit_config
   ```
 * For using the realsense camera you will need `lr_realsense`:
   ```
   git clone --branch crzz-dev https://gitlab.idiap.ch/learn-real/realsense.git lr_realsense
   ```

You will also need to install some python3 modules, preferably in a python virtual
environment. You can use the build_virtualenv.sh helper script in the lr_gym folder.
However you need to have python3.7 (stable_baselines 2 requires tensorflow 1.15, which requires
python<=3.7).

You can install python 3.7 with the following:

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.7 python3.7-venv python3.7-dev
```

At this point you can create the virtual python environment with the following helper
script (if you don't have a gpu you can specify sb_cpu instead of sb):

```
src/lr_gym/lr_gym/build_virtualenv.sh sb
```

At this point you can enter the virtual environment with:

```
. ./virtualenv/lr_gym_sb/bin/activate
```


You can then attempt at installing all the ros dependencies with rosdep (you may
 need to install python3-rosdep, *DO NOT* install python3-rosdep2):

```
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```


For building the workspace I use catkin build. To have catkin build
you can install catkin-tools which at the moment can be done with:

```
sudo apt install python3-catkin-tools python3-osrf-pycommon
```

You can now build with:

```
catkin build
```

To use in headless setups you may want to use pyvirtualdisplay, for it to work you
will need to install the following system dependencies:

```
sudo apt install xvfb xserver-xephyr tigervnc-standalone-server xfonts-base
```


## Examples - Cartpole

The environment can be tested using a python script that executes a hard-coded policy
through the python gym interface:

```
rosrun lr_gym test_cartpole_env.py
```

By default the script does not use the simulated camera, it is possible to enable
it with:

```
rosrun lr_gym test_cartpole_env.py --render
```

The rendered frames can be saved to file by specifying the --saveframes option.
The simulation step length can be changed using the --steplength option (the default is 0.05s)


### Basic Training

It is also possible to train a basic DQN policy by using the following:

```
rosrun lr_gym solve_cartpole.py
```

To see the simulation you can launch a gazebo client using the following:

```
roslaunch lr_gym gazebo_client.launch id:=0
```

It should reach the maximum episode length of 500 steps in about 400 episodes.


### Parallel Simulations Training with SAC

You can also train a SAC policy using multiple Gazebo simulations at the same time.

```
rosrun lr_gym solve_cartpole_sb2_sac_vec.py --envsNum=4
```

This will start 4 gazebo simulations in 4 different ros masters. To view the simulations
you can launch the following changing the id parameter:

```
roslaunch lr_gym gazebo_client.launch id:=0
```

It should reach the maximum episode length  of 500 steps in about 60 episode batches (60x4 episodes)


## Example - Hopper

You can execute a pre-trained policy with:
```
rosrun lr_gym solve_hopper.py --load src/lr_gym/lr_gym/trained_models/td3_hopper_20200605-175927s140000gazebo-urdf-gazeboGym.zip
```

You can train a new TD3 policy with

```
rosrun lr_gym solve_hopper.py
```

You can specify the number of timesteps to train for with the --iterations option.

You can also run the hopper environment in pybullet specifying the --pybullet option, in this mode you can also use an mjcf model (adapted from the OpenAI gym one) instead of the urdf one, using the --mjcf option. Trained models are also provided for these two modes.

As always the simulation can be visualized using 'roslaunch lr_gym gazebo_client.launch'

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

Scripts that solve or test each of these evironments in different ways are available in the examples folder. PandaEffortKeepVarPose and PandaMoveiPick currently do not have a working solve script.

You can try two pretrained models with the following:
 * PandaMoveitVarReachingEnv:
   ```
    rosrun lr_gym solve_pandaMoveitVarReaching.py --load src/lr_gym/lr_gym/trained_models/pandaMoveitPoseReachingEnv320210309-184658s200000_62400_steps.zip
   ```
 * PandaEffortKeepPose:
   ```
    rosrun lr_gym solve_panda_effort_keep_pose_vec.py --load solve_panda_effort_keep_tensorboard/20201114-212527/checkpoints/sac_pandaEffortKeep_20201114-212527s15000_3000000_steps.zip
   ```

You can test the PandaMoveitPick environment with an hard-coded policy using the test_pandaMoveitPick.py example.
* **In simulation**:
  ```
  rosrun lr_gym test_pandaMoveitPick.py
  ```
* **In the real** (substituting your robot ip):
  ```
  rosrun lr_gym test_pandaMoveitPick.py --real --robot_ip x.x.x.x
  ```

The `--real --robot_ip x.x.x.x` options can also be used for the PandaMoveitVarReachingEnv and PandaMoveitReachingEnv environments, no full training has been attempted.

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
The repository was tested on one laptop:

1. An Alienware laptop equipped with an Intel i7-6820HK CPU and an Nvidia GeForce GTX 980M GPU.

Here is reported the performance of the test_cartpole_env script.

| Laptop |       No Rendering        | With Rendering            |
|--------|---------------------------|---------------------------|
|   1    |  459fps (20.1x real time) | 188.1fps (8.86x real time) |

The simulation step was kept at 0.05s for all the tests.
