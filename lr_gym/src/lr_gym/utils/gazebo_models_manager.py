from importlib.util import module_for_loader
from pkg_resources import require
from lr_gym.utils.utils import Pose

from typing import List, Dict
import os
import subprocess
from gazebo_msgs.srv import SpawnModelRequest,SpawnModel
import rospy
from gazebo_msgs.srv import DeleteModel, DeleteModelRequest
import lr_gym.utils.dbg.ggLog as ggLog
import time

spawned_models = []

def waitService(servicename, serviceclass):
    while True:
        try:
            rospy.wait_for_service(servicename, timeout=10)
            break
        except rospy.ROSException as e:
            ggLog.info(f"Waiting for service {servicename}... (got error {e})")
            time.sleep(1)

    return rospy.ServiceProxy(servicename, serviceclass)


def compile_xacro(xacro_file_path : str, args : Dict[str,str]):
    args_str = []
    for k,v in args.items():
        args_str.append(f"{k}:={v} ")
    try:
        compiled_urdf = subprocess.check_output(["xacro", xacro_file_path]+args_str)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Xacro compilation failed with error {e}")
    return compiled_urdf.decode("utf-8") 


def spawn_model(xacro_file_path : str,
                pose : Pose = Pose(0,0,0,0,0,0,1), 
                args : Dict[str,str] = {}, 
                model_name = "model", 
                robot_namespace = "", 
                reference_frame = "world"):
    urdf_string = compile_xacro(xacro_file_path,args)
    gazebo_namespace = "gazebo"
    spawn_urdf_model = waitService(gazebo_namespace+'/spawn_urdf_model', SpawnModel)

    request = SpawnModelRequest()
    request.model_name = model_name
    request.model_xml = urdf_string
    request.robot_namespace = robot_namespace
    request.initial_pose = pose.getPoseStamped(reference_frame).pose
    request.reference_frame = reference_frame
    
    response = spawn_urdf_model.call(request)

    if not response.success:
        raise RuntimeError(f"Failed to spawn model {xacro_file_path} with args {args}, response:\n  {response}")

    spawned_models.append(model_name)

def delete_model(model_name : str):
    request = DeleteModelRequest()
    request.model_name = model_name

    gazebo_namespace = "gazebo"
    delete_model = waitService(gazebo_namespace+'/delete_model', DeleteModel)
    response = delete_model.call(request)

    if not response.success:
        raise RuntimeError(f"Failed to delete model {model_name} response:\n  {response}")

    spawned_models.remove(model_name)

def delete_all_models():
    while len(spawned_models)>0:
        delete_model(spawned_models[-1])