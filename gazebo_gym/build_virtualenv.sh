#!/bin/bash

ls ./src/
if [ $? -ne 0 ]; then
    echo "Couldn't find src folder. Are you in the root of a catkin workspace? You should be"
    exit 1
fi

if [ $# -ne 1 ]; then
    echo "You must specify the venv type (tf|sb|sb3)"
    exit 1
fi

venv_type="$1"
venv_name="gazebo_gym_$venv_type"

mkdir virtualenv
cd virtualenv
python3 -m venv $venv_name
. "$venv_name/bin/activate"
if [ $? -ne 0 ]; then
    echo "Failed to activate virtualenv"
    exit 1
fi

pip3 install pip --upgrade
pip3 install setuptools --upgrade
pip3 install wheel --upgrade

pip3 install -r "../src/gazebo_gym/gazebo_gym/requirements_$venv_type.txt"
