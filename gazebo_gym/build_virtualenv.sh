#!/bin/bash

ls ./src/
if [ $? -ne 0 ]; then
    echo "Couldn't find src folder. Are you in the root of a catkin workspace? You should be"
    exit 1
fi

mkdir virtualenv
cd virtualenv
python3 -m venv gazebo_gym
. gazebo_gym/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtualenv"
    exit 1
fi

pip3 install pip --upgrade
pip3 install setuptools --upgrade

pip3 install -r ../src/gazebo_gym/gazebo_gym/requirements.txt
