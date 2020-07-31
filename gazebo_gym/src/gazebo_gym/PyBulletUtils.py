
import pybullet as p
import pybullet_data


def start():
    """
    Start Pyullet simulation.

    This ends up calling examples/SharedMemory/PhysicsServerCommandProcessor.cpp:createEmptyDynamicsWorld()
    This means it uses a MultiBodyDynamicsWorld
    """
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

def buildPlaneWorld():
    # Taken from pybullet's scene_abstract.py
    p.setGravity(0, 0, -9.8)
    p.setDefaultContactERP(0.9)
    #print("self.numSolverIterations=",self.numSolverIterations)
    p.setPhysicsEngineParameter(fixedTimeStep=0.0165 / 4 * 4,
                                numSolverIterations=5,
                                numSubSteps=4)

    print("Physics engine parameters:"+str(p.getPhysicsEngineParameters()))

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeObjId = p.loadURDF("plane.urdf")

    # Taken from pybullet's scene_stadium.py
    p.changeDynamics(planeObjId, -1, lateralFriction=0.8, restitution=0.5)

    #Taken from env_bases.py (works both with and without)
    # p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

    return planeObjId

def _setupBody(bodyId : int) -> None:
    #p.changeDynamics(bodyId, -1, linearDamping=0, angularDamping=0)

    for j in range(p.getNumJoints(bodyId)):
        #p.changeDynamics(bodyId, j, linearDamping=0, angularDamping=0)
        p.setJointMotorControl2(bodyId,
                                j,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=0,
                                targetVelocity=0,
                                positionGain=0.1,
                                velocityGain=0.1,
                                force=0)
        print("Joint "+str(j)+" dynamics info: "+str(p.getDynamicsInfo(bodyId,j)))


def loadModel(modelFilePath : str, fileFormat : str = "urdf"):
    if fileFormat == "urdf":
        objId = p.loadURDF(modelFilePath, flags=p.URDF_USE_SELF_COLLISION)
    elif fileFormat == "mjcf":
        objId = p.loadMJCF(modelFilePath, flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)[0]
    else:
        raise AttributeError("Invalid format "+str(fileFormat))

    _setupBody(objId)

def buildSimpleEnv(modelFilePath : str, fileFormat : str = "urdf"):
    start()
    print("Started pybullet")
    buildPlaneWorld()
    print("Loading model "+str(modelFilePath))
    loadModel(modelFilePath, fileFormat)
    print("Loaded model")
