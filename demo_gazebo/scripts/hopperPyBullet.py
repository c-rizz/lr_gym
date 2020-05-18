import pybullet as p
import time

import pybullet_data
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--manualControl", default=False, action='store_true', help="Control joints manually")
ap.add_argument("--sleepLength", required=False, default=0.001, type=float, help="Sleep length in the main cycle")
ap.set_defaults(feature=True)
args = vars(ap.parse_args())

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeID = p.loadURDF("plane.urdf")
obUids = p.loadURDF(os.path.dirname(os.path.realpath(__file__))+"/../models/hopper.urdf")
humanoid = obUids

gravId = p.addUserDebugParameter("gravity", -10, 10, -10)
jointIds = []
paramIds = []

p.setPhysicsEngineParameter(numSolverIterations=10)
p.changeDynamics(humanoid, -1, linearDamping=0, angularDamping=0)
p.setTimeStep(0.001)

for j in range(p.getNumJoints(humanoid)):
    p.changeDynamics(humanoid, j, linearDamping=0, angularDamping=0)
    p.setJointMotorControl2(humanoid,
                            j,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=0,
                            targetVelocity=0,
                            positionGain=0.1,
                            velocityGain=0.1,
                            force=0)
if args["manualControl"]:
    for j in [3,4,5]:
        info = p.getJointInfo(humanoid, j)
        #print(info)
        jointName = info[1]
        jointType = info[2]
        if jointType == p.JOINT_REVOLUTE:
            jointIds.append(j)
            paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, 0))

while (1):
    p.setGravity(0, 0, p.readUserDebugParameter(gravId))
    for i in range(len(paramIds)):
        c = paramIds[i]
        targetPos = p.readUserDebugParameter(c)
        p.setJointMotorControl2(humanoid, jointIds[i], p.POSITION_CONTROL, targetPos, force=5 * 240.)
    p.stepSimulation()
    #print("stepped")
    time.sleep(args["sleepLength"])
