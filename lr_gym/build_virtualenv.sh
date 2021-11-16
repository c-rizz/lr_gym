#!/bin/bash

ls ./src/
if [ $? -ne 0 ]; then
    echo "Couldn't find src folder. Are you in the root of a catkin workspace? You should be"
    exit 1
fi

if [ $# -ne 1 ]; then
    echo "You must specify the venv type (tf|sb|sb3|sb_cpu)"
    exit 1
fi

venv_type="$1"
venv_name="lr_gym_$venv_type"

mkdir virtualenv
cd virtualenv
echo "Creating venv..."
python -m venv $venv_name #Tensorflow 1.15 requires python<=3.7
echo "Done."
echo "Activating venv..."
. "$venv_name/bin/activate"
if [ $? -ne 0 ]; then
    echo "Failed to activate virtualenv"
    exit 1
fi
echo "Done."

echo "Installing modules.."
python -m pip install pip --upgrade
python -m pip install setuptools --upgrade
python -m pip install wheel --upgrade

python -m pip install -r "../src/lr_gym/lr_gym/requirements_$venv_type.txt"
echo "Done."
