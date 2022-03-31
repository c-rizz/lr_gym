import typing
from nptyping import NDArray
import numpy as np
import sensor_msgs
import geometry_msgs
import sensor_msgs.msg
import time
import cv2
import collections
from typing import List, Tuple, Callable, Dict, Union
import os
import quaternion
import signal
import datetime
import inspect
import shutil
import tqdm
from pathlib import Path
import random
import multiprocessing
import csv
from pynvml.smi import nvidia_smi
import yaml
import torch as th

import lr_gym.utils.dbg.ggLog as ggLog

name_to_dtypes = {
    "rgb8":    (np.uint8,  3),
    "rgba8":   (np.uint8,  4),
    "rgb16":   (np.uint16, 3),
    "rgba16":  (np.uint16, 4),
    "bgr8":    (np.uint8,  3),
    "bgra8":   (np.uint8,  4),
    "bgr16":   (np.uint16, 3),
    "bgra16":  (np.uint16, 4),
    "mono8":   (np.uint8,  1),
    "mono16":  (np.uint16, 1),

    # for bayer image (based on cv_bridge.cpp)
    "bayer_rggb8":  (np.uint8,  1),
    "bayer_bggr8":  (np.uint8,  1),
    "bayer_gbrg8":  (np.uint8,  1),
    "bayer_grbg8":  (np.uint8,  1),
    "bayer_rggb16": (np.uint16, 1),
    "bayer_bggr16": (np.uint16, 1),
    "bayer_gbrg16": (np.uint16, 1),
    "bayer_grbg16": (np.uint16, 1),

    # OpenCV CvMat types
    "8UC1":    (np.uint8,   1),
    "8UC2":    (np.uint8,   2),
    "8UC3":    (np.uint8,   3),
    "8UC4":    (np.uint8,   4),
    "8SC1":    (np.int8,    1),
    "8SC2":    (np.int8,    2),
    "8SC3":    (np.int8,    3),
    "8SC4":    (np.int8,    4),
    "16UC1":   (np.uint16,   1),
    "16UC2":   (np.uint16,   2),
    "16UC3":   (np.uint16,   3),
    "16UC4":   (np.uint16,   4),
    "16SC1":   (np.int16,  1),
    "16SC2":   (np.int16,  2),
    "16SC3":   (np.int16,  3),
    "16SC4":   (np.int16,  4),
    "32SC1":   (np.int32,   1),
    "32SC2":   (np.int32,   2),
    "32SC3":   (np.int32,   3),
    "32SC4":   (np.int32,   4),
    "32FC1":   (np.float32, 1),
    "32FC2":   (np.float32, 2),
    "32FC3":   (np.float32, 3),
    "32FC4":   (np.float32, 4),
    "64FC1":   (np.float64, 1),
    "64FC2":   (np.float64, 2),
    "64FC3":   (np.float64, 3),
    "64FC4":   (np.float64, 4)
}


def image_to_numpy(rosMsg : sensor_msgs.msg.Image) -> np.ndarray:
    """Extracts an numpy/opencv image from a ros sensor_msgs image

    Parameters
    ----------
    rosMsg : sensor_msgs.msg.Image
        The ros image message

    Returns
    -------
    np.ndarray
        The numpy array contaning the image. Compatible with opencv

    Raises
    -------
    TypeError
        If the input image encoding is not supported

    """
    if rosMsg.encoding not in name_to_dtypes:
        raise TypeError('Unrecognized encoding {}'.format(rosMsg.encoding))

    dtype_class, channels = name_to_dtypes[rosMsg.encoding]
    dtype = np.dtype(dtype_class)
    dtype = dtype.newbyteorder('>' if rosMsg.is_bigendian else '<')
    shape = (rosMsg.height, rosMsg.width, channels)

    data = np.frombuffer(rosMsg.data, dtype=dtype).reshape(shape)
    data.strides = (
        rosMsg.step,
        dtype.itemsize * channels,
        dtype.itemsize
    )

    if not np.isfinite(data).all():
        ggLog.warn(f"image_to_numpy(): nan detected in image")



    # opencv uses bgr instead of rgb
    # probably should be done also for other encodings
    if rosMsg.encoding == "rgb8":
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

    if channels == 1:
        data = data[...,0]
    return data

def numpyImg_to_ros(img : np.ndarray) -> sensor_msgs.msg.Image:
    """
    """
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rosMsg = sensor_msgs.msg.Image()
    rosMsg.data = img.tobytes()
    rosMsg.step = img.strides[0]
    rosMsg.is_bigendian = (img.dtype.byteorder == '>')

    if img.shape[2] == 3:
        rosMsg.encoding = "rgb8"
    elif img.shape[2] == 1:
        rosMsg.encoding = "mono8"


class JointState:
    # These are lists because the joint may have multiple DOF
    position = []
    rate = []
    effort = []

    def __init__(self, position : List[float], rate : List[float], effort : List[float]):
        self.position = position
        self.rate = rate
        self.effort = effort

    def __str__(self):
        return "JointState("+str(self.position)+","+str(self.rate)+","+str(self.effort)+")"

    def __repr__(self):
        return self.__str__()

class AverageKeeper:
    def __init__(self, bufferSize = 100):
        self._bufferSize = bufferSize
        self.reset()

    def addValue(self, newValue):
        self._buffer.append(newValue)
        self._avg = float(sum(self._buffer))/len(self._buffer)

    def getAverage(self):
        return self._avg

    def reset(self):
        self._buffer = collections.deque(maxlen=self._bufferSize)
        self._avg = 0

def buildPoseStamped(position_xyz, orientation_xyzw, frame_id):
    pose = geometry_msgs.msg.PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = position_xyz[0]
    pose.pose.position.y = position_xyz[1]
    pose.pose.position.z = position_xyz[2]
    pose.pose.orientation.x = orientation_xyzw[0]
    pose.pose.orientation.y = orientation_xyzw[1]
    pose.pose.orientation.z = orientation_xyzw[2]
    pose.pose.orientation.w = orientation_xyzw[3]
    return pose

def pyTorch_makeDeterministic():
    """ Make pytorch as deterministic as possible.
        Still, DOES NOT ENSURE REPRODUCIBILTY ACROSS DIFFERENT TORCH/CUDA BUILDS AND
        HARDWARE ARCHITECTURES
    """
    import torch as th
    th.manual_seed(0)
    np.random.seed(0)
    th.backends.cudnn.benchmark = False
    th.use_deterministic_algorithms(True)
    # Following may make things better, see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def quaternionDistance(q1 : quaternion.quaternion ,q2 : quaternion.quaternion ):
    """ Returns the minimum angle that separates two orientations.

    Parameters
    ----------
    q1 : quaternion.quaternion
        Description of parameter `q1`.
    q2 : quaternion.quaternion
        Description of parameter `q2`.

    Returns
    -------
    type
        Description of returned object.

    Raises
    -------
    ExceptionName
        Why the exception is raised.

    """
    # q1a = quaternion.as_float_array(q1)
    # q2a = quaternion.as_float_array(q2)
    #
    # return np.arccos(2*np.square(np.inner(q1a,q2a)) - 1)
    return quaternion.rotation_intrinsic_distance(q1,q2)

def buildQuaternion(x,y,z,w):
    return quaternion.quaternion(w,x,y,z)

class Pose:
    position : NDArray[(3,), np.float32]
    orientation : np.quaternion

    def __init__(self, x,y,z, qx,qy,qz,qw):
        self.position = np.array([x,y,z])
        self.orientation = buildQuaternion(x=qx,y=qy,z=qz,w=qw)

    def __str__(self):
        return f"[{self.position[0],self.position[1],self.position[2],self.orientation.x,self.orientation.y,self.orientation.z,self.orientation.w}]"

    def getPoseStamped(self, frame_id : str):
        return buildPoseStamped(self.position, np.array([self.orientation.x,self.orientation.y,self.orientation.z,self.orientation.w]), frame_id=frame_id)

    def getListXyzXyzw(self):
        return [self.position[0],self.position[1],self.position[2],self.orientation.x,self.orientation.y,self.orientation.z,self.orientation.w]
        
    def __repr__(self):
        return self.__str__()

class LinkState:
    pose : Pose = None
    pos_velocity_xyz : Tuple[float, float, float] = None
    ang_velocity_xyz : Tuple[float, float, float] = None

    def __init__(self, position_xyz : Tuple[float, float, float], orientation_xyzw : Tuple[float, float, float, float],
                    pos_velocity_xyz : Tuple[float, float, float], ang_velocity_xyz : Tuple[float, float, float]):
        self.pose = Pose(position_xyz[0],position_xyz[1],position_xyz[2], orientation_xyzw[0],orientation_xyzw[1],orientation_xyzw[2],orientation_xyzw[3])
        self.pos_velocity_xyz = pos_velocity_xyz
        self.ang_velocity_xyz = ang_velocity_xyz

    def __str__(self):
        return "LinkState("+str(self.pose)+","+str(self.pos_velocity_xyz)+","+str(self.ang_velocity_xyz)+")"

    def __repr__(self):
        return self.__str__()


did_initialize_sigint_handling = False
sigint_received = False
sigint_counter = 0
sigint_max = 10
original_sigint_handler = None

def setupSigintHandler():
    global original_sigint_handler
    global did_initialize_sigint_handling
    if original_sigint_handler is None:
        original_sigint_handler = signal.getsignal(signal.SIGINT)

    def sigint_handler(signal, stackframe):
        global sigint_received
        global sigint_counter
        global sigint_max
        global original_sigint_handler
        sigint_received = True
        sigint_counter += 1
        print(f"\n"+
              f"-----------------------------------------------------------------------------------------------------\n"+
              f"-----------------------------------------------------------------------------------------------------\n"+
              f"Received sigint, will halt at first opportunity. ({sigint_max-sigint_counter} presses to hard SIGINT)\n"+
              f"-----------------------------------------------------------------------------------------------------\n"+
              f"-----------------------------------------------------------------------------------------------------\n\n")
        if sigint_counter>sigint_max:
            try:
                original_sigint_handler(signal,stackframe)
            except KeyboardInterrupt:
                pass #If it was the original one, doesn't do anything, if it was something else it got executed
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, sigint_handler)
    did_initialize_sigint_handling = True

def haltOnSigintReceived():
    if not did_initialize_sigint_handling:
        return
    global sigint_received
    global sigint_counter
    if sigint_received:
        answer = input(f"SIGINT received. Press Enter to resume or type 'exit' to terminate:\n> ")
        if answer == "exit":
            original_sigint_handler(signal.SIGINT, None)
            raise KeyboardInterrupt
        print("Resuming...")
        sigint_received = False
        sigint_counter = 0



def createSymlink(src, dst):
    try:
        os.symlink(src, dst)
    except FileExistsError:
        try:
            os.unlink(dst)
            time.sleep(random.random()*10) #TODO: have a better way to avoid race conditions
            os.symlink(src, dst)
        except:
            pass

def setupLoggingForRun(file : str, currentframe = None, run_id_prefix : str = "", folderName : str = None):
    """Sets up a logging output folder for a training run.
        It creates the folder, saves the current main script file for reference

    Parameters
    ----------
    file : str
        Path of the main script, the file wil be copied in the log folder
    frame : [type]
        Current frame from the main method, use inspect.currentframe() to get it. It will be used to save the
        call parameters.

    Returns
    -------
    str
        The logging folder to be used
    """
    if folderName is None:
        run_id = run_id_prefix+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        script_out_folder = os.getcwd()+"/"+os.path.basename(file)
        folderName = script_out_folder+"/"+run_id
        os.makedirs(folderName, exist_ok=True)
    else:
        os.makedirs(folderName, exist_ok=True)
        script_out_folder = str(Path(folderName).parent.absolute())

    createSymlink(src = folderName, dst = script_out_folder+"/latest")
    shutil.copyfile(file, folderName+"/main_script")
    if currentframe is not None:
        args, _, _, values = inspect.getargvalues(currentframe)
    else:
        args, values = ([],{})
    # inputargs = [(i, values[i]) for i in args]
    # with open(folderName+"/input_args.txt", "w") as input_args_file:
    #     print(str(inputargs), file=input_args_file)
    with open(folderName+"/input_args.yaml", "w") as input_args_yamlfile:
        yaml.dump(values,input_args_yamlfile, default_flow_style=None)
    return folderName


    
def lr_gym_startup(main_file_path : str, currentframe = None, using_pytorch : bool = True, run_id_prefix : str = "", folderName : str = None) -> str:
    logFolder = setupLoggingForRun(main_file_path, currentframe, run_id_prefix=run_id_prefix, folderName=folderName)
    ggLog.addLogFile(logFolder+"/gglog.log")
    setupSigintHandler()
    if using_pytorch:
        import torch as th
        pyTorch_makeDeterministic()
        # th.autograd.set_detect_anomaly(True) # Detect NaNs
        if th.cuda.is_available():
            ggLog.info(f"CUDA AVAILABLE: device = {th.cuda.get_device_name()}")
        else:
            ggLog.warn("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"+
                        "                  NO CUDA AVAILABLE!\n"+
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n"+
                        "Will continue in 10 sec...")
            time.sleep(10)
    return logFolder

def evaluatePolicy(env, model, episodes : int, on_ep_done_callback = None):

    rewards = np.empty((episodes,), dtype = np.float32)
    steps = np.empty((episodes,), dtype = np.int32)
    wallDurations = np.empty((episodes,), dtype = np.float32)
    predictWallDurations = np.empty((episodes,), dtype = np.float32)
    totDuration=0.0
    successes = 0.0
    #frames = []
    #do an average over a bunch of episodes
    for episode in tqdm.tqdm(range(0,episodes)):
        frame = 0
        episodeReward = 0
        done = False
        predDurations = []
        t0 = time.monotonic()
        # ggLog.info("Env resetting...")
        obs = env.reset()
        # ggLog.info("Env resetted")
        while not done:
            t0_pred = time.monotonic()
            # ggLog.info("Predicting")
            action, _states = model.predict(obs)
            predDurations.append(time.monotonic()-t0_pred)
            # ggLog.info("Stepping")
            obs, stepReward, done, info = env.step(action)
            frame+=1
            episodeReward += stepReward
            # ggLog.info(f"Step reward = {stepReward}")
        rewards[episode]=episodeReward
        if "success" in info.keys():
            if info["success"]:
                ggLog.info(f"Success {successes} ratio = {successes/(episode+1)}")
                successes += 1
        steps[episode]=frame
        wallDurations[episode]=time.monotonic() - t0
        predictWallDurations[episode]=sum(predDurations)
        if on_ep_done_callback is not None:
            on_ep_done_callback(episodeReward=episodeReward, steps=frame, episode=episode)
        ggLog.debug("Episode "+str(episode)+" lasted "+str(frame)+" frames, total reward = "+str(episodeReward))
    eval_results = {"reward_mean" : np.mean(rewards),
                    "reward_std" : np.std(rewards),
                    "steps_mean" : np.mean(steps),
                    "steps_std" : np.std(steps),
                    "success_ratio" : successes/episodes,
                    "wall_duration_mean" : np.mean(wallDurations),
                    "wall_duration_std" : np.std(wallDurations),
                    "predict_wall_duration_mean" : np.mean(predictWallDurations),
                    "predict_wall_duration_std" : np.std(predictWallDurations)}
    return eval_results

def fileGlobToList(fileGlobStr : str):
    """Convert a file path glob (i.e. a file path ending with *) to a list of files

    Parameters
    ----------
    fileGlobStr : str
        a string representing a path, possibly with an asterisk at the end

    Returns
    -------
    List
        A list of files
    """
    if fileGlobStr.endswith("*"):
        folderName = os.path.dirname(fileGlobStr)
        fileNamePrefix = os.path.basename(fileGlobStr)[:-1]
        files = []
        for f in os.listdir(folderName):
            if f.startswith(fileNamePrefix):
                files.append(f)
        files = sorted(files, key = lambda x: int(x.split("_")[-2]))
        fileList = [folderName+"/"+f for f in files]
        numEpisodes = 1
    else:
        fileList = [fileGlobStr]
    return fileList



def evaluateSavedModels(files : List[str], evaluator : Callable[[str],Dict[str,Union[float,int,str]]], maxProcs = int(multiprocessing.cpu_count()/2), args = []):
    # file paths should be in the format ".../<__file__>/<run_id>/checkpoints/<model.zip>"
    loaded_run_id = files[0].split("/")[-2]
    run_id = "eval_of_"+loaded_run_id+"_at_"+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    folderName = os.getcwd()+"/"+os.path.basename(__file__)+"/eval/"+run_id
    os.makedirs(folderName)
    csvfilename = folderName+"/evaluation.csv"
    with open(csvfilename,"w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter = ",")
        neverWroteToCsv = True

        processes = maxProcs
        print(f"Using {processes} parallel evaluators")
        argss = [[file,*args] for file in files]
        with multiprocessing.Pool(processes) as p:
            eval_results = p.map(evaluator, argss)

        for i in range(len(argss)):
            eval_results[i]["file"] = argss[i][0]

        if neverWroteToCsv:
            csvwriter.writerow(eval_results[0].keys())
            neverWroteToCsv = False
        for eval_results in eval_results:
            csvwriter.writerow(eval_results.values())
            csvfile.flush()


def getBestGpu():
    gpus_mem_info = []
    for i in range(th.cuda.device_count()):
        prevDev = th.cuda.current_device()
        th.cuda.set_device(th.device(type="cuda", index=i))
        gpus_mem_info.append(th.cuda.mem_get_info()) #Returns [free, total]
        th.cuda.set_device(prevDev)
        print(f"Got {gpus_mem_info[-1]}")

    bestRatio = 0
    bestGpu = None
    for i in range(len(gpus_mem_info)):
        tot = gpus_mem_info[i][1]
        free = gpus_mem_info[i][0]
        ratio = free/tot
        if ratio > bestRatio:
            bestRatio = ratio
            bestGpu = i
    ggLog.info(f"Choosing GPU {bestGpu} with {bestRatio*100}% free memory")
    return bestGpu


def torch_selectBestGpu():
    import torch as th
    bestGpu = getBestGpu()
    th.cuda.set_device(bestGpu)
    return th.device('cuda:'+str(bestGpu))
