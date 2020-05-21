import pybullet as p
import time

import pybullet_data
import os

from PyBulletController import PyBulletController


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeID = p.loadURDF("plane.urdf")
cartpoleObjId = p.loadURDF(os.path.dirname(os.path.realpath(__file__))+"/../models/cartpole_v0.urdf")

jointIds = []
paramIds = []

p.setPhysicsEngineParameter(numSolverIterations=10)
p.changeDynamics(cartpoleObjId, -1, linearDamping=0, angularDamping=0)
p.setTimeStep(0.001)

for j in range(p.getNumJoints(cartpoleObjId)):
    p.changeDynamics(cartpoleObjId, j, linearDamping=0, angularDamping=0)
    p.setJointMotorControl2(cartpoleObjId,
                            j,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=0,
                            targetVelocity=0,
                            positionGain=0.1,
                            velocityGain=0.1,
                            force=0)


pybulletController = PyBulletController()
