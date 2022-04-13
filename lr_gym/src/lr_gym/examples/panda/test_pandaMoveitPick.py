#!/usr/bin/env python3

import time
import numpy as np
import argparse
from datetime import datetime

from lr_gym.envs.PandaMoveitPickEnv import PandaMoveitPickEnv
from lr_gym.envs.GymEnvWrapper import GymEnvWrapper
import lr_gym.utils.dbg.ggLog as ggLog
from lr_gym.envControllers.MoveitRosController import MoveitRosController


def main(real : bool, robot_ip : str) -> None:
    """Solve the gazebo Panda reaching environment."""


    #rospy.init_node('solve_panda_reaching_moveit', anonymous=True, log_level=rospy.WARN)
    #env = gym.make('CartPoleStayUp-v0')

    runId = datetime.now().strftime('%Y%m%d-%H%M%S')
    folderName = "./test_pandaPick/"+runId

    print("Setting up environment...")
    if not real:
        ggEnv = PandaMoveitPickEnv( #goalPose=[0.3,-0.3,0.5,-1,0,0,0],
                                    maxStepsPerEpisode = 30,
                                    backend="gazebo")
    else:
        environmentController = MoveitRosController(jointsOrder =  [("panda","panda_joint1"),
                                                                    ("panda","panda_joint2"),
                                                                    ("panda","panda_joint3"),
                                                                    ("panda","panda_joint4"),
                                                                    ("panda","panda_joint5"),
                                                                    ("panda","panda_joint6"),
                                                                    ("panda","panda_joint7")],
                                                    endEffectorLink  = ("panda", "panda_tcp"),
                                                    referenceFrame   = "world",
                                                    initialJointPose = {("panda","panda_joint1") : 0,
                                                                        ("panda","panda_joint2") : 0,
                                                                        ("panda","panda_joint3") : 0,
                                                                        ("panda","panda_joint4") :-1,
                                                                        ("panda","panda_joint5") : 0,
                                                                        ("panda","panda_joint6") : 1,
                                                                        ("panda","panda_joint7") : 3.14159/4},
                                                    gripperActionTopic = "/franka_gripper/gripper_action",
                                                    gripperInitialWidth = 0.08)
        ggEnv = PandaMoveitPickEnv( #goalPose=[0.3,-0.3,0.5,-1,0,0,0],
                                    maxStepsPerEpisode = 30,
                                    backend="real",
                                    real_robot_ip=robot_ip,
                                    environmentController = environmentController)

    env = GymEnvWrapper(ggEnv,
                        episodeInfoLogFile = folderName+"/GymEnvWrapper_log.csv")
    print("Environment created")

    #setup seeds for reproducibility
    RANDOM_SEED=20200730
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)

    time.sleep(5) #Wait so the console output is cleaner (Just for visualization purposes)
    input("Press Enter to continue...")

    print("Testing...")
    


    actionseq = [   #Open and close
                    [ 0.0,  0.0,  0.0,    0.0,  0.0,  0.0,    1.0,0.5],
                    [ 0.0,  0.0,  0.0,    0.0,  0.0,  0.0,    0.0,0.5],
                    [ 0.0,  0.0,  0.0,    0.0,  0.0,  0.0,    1.0,0.5],
                    ##Move around a bit
                    #[-1.0,  0.0,  0.0,    0.0,  0.0,  0.0,    1.0,0.5],
                    #[ 1.0,  0.0,  0.0,    0.0,  0.0,  0.0,    1.0,0.5],
                    #[ 0.0,  0.0,  0.0,    1.0,  0.0,  0.0,    1.0,0.5],
                    #[ 0.0,  0.0,  0.0,   -1.0,  0.0,  0.0,    1.0,0.5],
                    #Go down
                    [ 0.0,  0.0, -1.0,    0.0,  0.0,  0.0,    1.0,0.5],
                    [ 0.0,  0.0, -1.0,    0.0,  0.0,  0.0,    1.0,0.5],
                    [ 0.0,  0.0, -1.0,    0.0,  0.0,  0.0,    1.0,0.5],
                    [ 0.0,  0.0, -1.0,    0.0,  0.0,  0.0,    1.0,0.5],
                    [ 0.0,  0.0, -1.0,    0.0,  0.0,  0.0,    1.0,0.5],
                    [ 0.0,  0.0, -1.0,    0.0,  0.0,  0.0,    1.0,0.5],
                    [ 0.0,  0.0, -0.8,    0.0,  0.0,  0.0,    1.0,0.5],
                    #Close
                    [ 0.0,  0.0,  0.0,    0.0,  0.0,  0.0,    0.0,0.5],
                    #Go up
                    [ 0.0,  0.0,  1.0,    0.0,  0.0,  0.0,    0.0,0.5],
                    [ 0.0,  0.0,  1.0,    0.0,  0.0,  0.0,    0.0,0.5],
                    [ 0.0,  0.0,  1.0,    0.0,  0.0,  0.0,    0.0,0.5],
                    [ 0.0,  0.0,  1.0,    0.0,  0.0,  0.0,    0.0,0.5],
                    [ 0.0,  0.0,  1.0,    0.0,  0.0,  0.0,    0.0,0.5],
                    [ 0.0,  0.0,  1.0,    0.0,  0.0,  0.0,    0.0,0.5],
                    [ 0.0,  0.0,  1.0,    0.0,  0.0,  0.0,    0.0,0.5],
                    [ 0.0,  1.0,  0.0,    0.0,  0.0,  0.0,    0.0,0.5],
                    #Open
                    [ 0.0,  0.0,  0.0,    0.0,  0.0,  0.0,    10.,0.5],
                    ]


    t_preVal = time.time()
    rewards=[]
    totFrames=0
    totDuration=0
    #frames = []
    #do an average over a bunch of episodes
    episode = 0
    #for episode in tqdm.tqdm(range(0,5000000)):
    while True:
        print("Episode "+str(episode))
        frame = 0
        episodeReward = 0
        done = False
        obs = env.reset()
        t0 = time.time()
        while frame < len(actionseq) and not done:
            #print("Episode "+str(episode)+" frame "+str(frame))
            action = actionseq[frame]
            #print("action = "+str(action))
            obs, stepReward, done, info = env.step(action)
            #frames.append(env.render("rgb_array"))
            frame+=1
            episodeReward += stepReward
            print("stepReward = ",stepReward)
            time.sleep(1)
        rewards.append(episodeReward)
        totFrames +=frame
        totDuration += time.time() - t0
        episode+=1
        #print("Episode "+str(episode)+" lasted "+str(frame)+" frames, total reward = "+str(episodeReward))
    avgReward = sum(rewards)/len(rewards)
    duration_val = time.time() - t_preVal
    print("Computed average reward. Took "+str(duration_val)+" seconds ("+str(totFrames/totDuration)+" fps).")
    print("Average reward = "+str(avgReward))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=False, action='store_true', help="Use the real robot")
    ap.add_argument("--robot_ip", default="0.0.0.0", type=str, help="Ip address of the robot")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())
    main(real=args["real"], robot_ip=args["robot_ip"])
