#!/usr/bin/env python3

import rospy
import errno
import time
import tqdm
import cv2
import os
import argparse
import lr_gym.utils.PyBulletUtils as PyBulletUtils
import stable_baselines3
from pyvirtualdisplay import Display

from lr_gym.envs.CartpoleEnv import CartpoleEnv
from lr_gym.envs.GymEnvWrapper import GymEnvWrapper
from lr_gym.envControllers.GazeboController import GazeboController
from lr_gym.envControllers.PyBulletController import PyBulletController


def main(simulator : str, doRender : bool = False, noPlugin : bool = False, saveFrames : bool = False, stepLength_sec : float = 0.05, sleepLength : float = 0, parallelEnvsNum : int = 1) -> None:
    """Run the gazebo cartpole environment with a simple hard-coded policy.

    Parameters
    ----------
    doRender : bool
        Set to True to enable the rendering of each simulation step
    noPlugin : bool
        set to True to disable the use of the gazebo lr_gym_env plugin
    saveFrames : bool
        Set to true to save every computed frame rendering to file. (will save to ./frames/)
    stepLength_sec : float
        Here you can set the duration in seconds of each simulation step

    Returns
    -------
    None

    """

    def constructEnv():
        if simulator == 'pybullet':
            PyBulletUtils.buildSimpleEnv(os.path.dirname(os.path.realpath(__file__))+"/../models/cartpole_v0.urdf")
            simulatorController = PyBulletController(stepLength_sec = stepLength_sec)
        if simulator == 'gazebo':
            simulatorController = GazeboController(stepLength_sec = stepLength_sec)
        return GymEnvWrapper(CartpoleEnv(render = doRender, stepLength_sec=stepLength_sec, simulatorController=simulatorController, startSimulation = True), quiet=True)
    env = stable_baselines3.common.vec_env.SubprocVecEnv([constructEnv for i in range(parallelEnvsNum)])
    
    #env = gym.make('CartPoleStayUp-v0')
    #env = CartpoleEnv(render = doRender, stepLength_sec=stepLength_sec, simulatorController=simulatorController, startSimulation = True)
    #setup seeds for reproducibility
    RANDOM_SEED=20200401
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    env._max_episode_steps = 500 #limit episode length

    imagesOutFolder = "./frames"

    createFolders(imagesOutFolder)

    rospy.loginfo("Testing with hardcoded policy")
    wallTimeStart = time.time()
    rewards=[]
    totFrames=0
    #frames = []
    totalSimTime = 0


    obss = env.reset()
    episodeCounter = 0
    #do an average over a bunch of episodes
    for step_count in tqdm.tqdm(range(0,1000)):
        #rospy.loginfo("resetting...")
        #rospy.loginfo("resetted")

        #rospy.loginfo("---------------------------------------")
        #time.sleep(1)
        #rospy.loginfo("Episode "+str(episode)+" frame "+str(frame))

        if doRender:
            img = env.render(mode = 'rgb_array')
            if saveFrames and img.size!=0:
                r = cv2.imwrite(imagesOutFolder+"/frame-"+str(step_count)+".png",img)
                if not r:
                    print("couldn't save image")
                #else:
                #    print("saved image")

        #rospy.loginfo(obs)
        action = [None]*len(obss)
        for i in range(len(obss)):
            if obss[i][2]>0:
                action[i] = 1
            else:
                action[i] = 0
        #rospy.loginfo("stepping("+str(action)+")...")
        obss, stepRewards, dones, infos = env.step(action)
        #rospy.loginfo("stepped")
        #frames.append(env.render("rgb_array"))

        episodeCounter += sum(dones)

        if sleepLength>0:
            time.sleep(sleepLength)
        step_count+=1

        totalSimTime += sum([info["simTime"] for info in infos])
        [rewards.append(stepReward) for stepReward in stepRewards]
        totFrames += 1
        #print("Episode "+str(episode)+" lasted "+str(frame)+" frames, total reward = "+str(episodeReward))
    avgReward = sum(rewards)/episodeCounter
    totalWallTime = time.time() - wallTimeStart

    print("Average reward is "+str(avgReward)+". Took "+str(totalWallTime)+" seconds ({:.3f}".format(totFrames/totalWallTime)+"x"+str(env.num_envs)+" fps). simTime/wallTime={:.3f}".format(totalSimTime/totalWallTime)+" total frames count = "+str(totFrames)+" episode count = "+str(episodeCounter))

    env.close()

def createFolders(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default=False, action='store_true', help="Enable camera rendering")
    ap.add_argument("--pybullet", default=False, action='store_true', help="Use pybullet simulator")
    ap.add_argument("--noplugin", default=False, action='store_true', help="Don't use the gazebo lr_gym_env plugin")
    ap.add_argument("--saveframes", default=False, action='store_true', help="Saves each frame of each episode in ./frames")
    ap.add_argument("--steplength", required=False, default=0.05, type=float, help="Duration of each simulation step")
    ap.add_argument("--sleeplength", required=False, default=0, type=float, help="How much to sleep at the end of each frame execution")
    ap.add_argument("--envsNum", required=False, default=1, type=int, help="Number of environments to run in parallel")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())

    with Display() as disp:
        if args["pybullet"]:
            simName = 'pybullet'
        else:
            simName = 'gazebo'


        main(   simulator = simName,
                doRender = args["render"],
                noPlugin=args["noplugin"],
                saveFrames=args["saveframes"],
                stepLength_sec=args["steplength"],
                sleepLength = args["sleeplength"],
                parallelEnvsNum = args["envsNum"])
