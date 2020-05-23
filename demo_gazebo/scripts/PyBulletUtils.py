
import pybullet as p
import typing
import pybullet_data
import os


def start():
    p.connect(p.GUI)

def buildPlaneWorld():
    p.setGravity(0,0,-9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeObjId = p.loadURDF("plane.urdf")
    return planeObjId

def loadModel(modelFilePath : str):
    #objId = p.loadURDF(modelFilePath)
    objId = p.loadMJCF(os.path.dirname(os.path.realpath(__file__))+"/../models/hopper.xml")[0]

    p.setPhysicsEngineParameter(numSolverIterations=10)
    p.changeDynamics(objId, -1, linearDamping=0, angularDamping=0)
    p.setTimeStep(0.0041666) #240fps

    for j in range(p.getNumJoints(objId)):
        p.changeDynamics(objId, j, linearDamping=0, angularDamping=0)
        p.setJointMotorControl2(objId,
                                j,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=0,
                                targetVelocity=0,
                                positionGain=0.1,
                                velocityGain=0.1,
                                force=0)

def buildSimpleEnv(modelFilePath : str):
    start()
    print("Started pybullet")
    buildPlaneWorld()
    print("Built world")
    loadModel(modelFilePath)
    print("Loaded model")
