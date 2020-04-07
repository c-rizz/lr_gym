#!/bin/bash

#find the library path and add it to GAZEBO_PLUGIN_PATH
libPath=$(echo $CMAKE_PREFIX_PATH | cut -d: -f1)/lib
export GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}:$(realpath $(echo $CMAKE_PREFIX_PATH | cut -d: -f1)/lib)
roslaunch gazebo_gym_env_plugin emptyWorld.launch
