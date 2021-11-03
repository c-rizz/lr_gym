#!/bin/bash

if [ -z ${1+x} ]; then
	echo "syntax is: install_helper.sh <dest_folder>"
	exit 0
fi

echo "This is meant only as a helper script, I do not guarantee this will work and will not have unexpected consequences."
echo "Before using please check what the script does, it should be quite straightforward."
echo "RUN THIS ONLY IF YOU KNOW WHAT YOU ARE DOING"
echo "Do you want to continue? [y/N]"
read ans

if [ $ans != "y" ]; then
    echo "exiting without installing."
    exit 0
fi



echo "Preparing workspace in $1 ..."
sleep 3

cd $1

mkdir src
cd src

git clone --branch a2110 https://github.com/c-rizz/lr_gym.git
git clone --branch a2110 https://github.com/c-rizz/lr_panda.git
git clone --branch a2110 https://github.com/c-rizz/lr_panda_moveit_config.git
git clone --branch a2110 https://github.com/c-rizz/lr_realsense.git


echo "Installing python 3.7..."
sleep 3
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get -y update
sudo apt-get -y install python3.7 python3.7-venv python3.7-dev

cd $1

echo "Creating python virtualenv..."
sleep 3
src/lr_gym/lr_gym/build_virtualenv.sh sb3
. ./virtualenv/lr_gym_sb/bin/activate

echo "Installing dependencies..."
sleep 3
sudo apt-get -y install python3-rosdep
rosdep update
rosdep install --from-paths src --ignore-src -r -y

sudo apt-get -y install python3-catkin-tools python3-osrf-pycommon
sudo apt-get -y install xvfb xserver-xephyr tigervnc-standalone-server xfonts-base

cd $1
echo "Building workspace in $1 ..."
sleep 3
. /opt/ros/noetic/setup.bash
catkin build -DCMAKE_BUILD_TYPE=Release

