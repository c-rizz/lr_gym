from collections.abc import Callable
import multiprocessing
from multiprocessing import Pipe
import cloudpickle
import pickle
from typing import Sequence, Dict, Any, Tuple
import lr_gym
import numpy as np

import os
import signal
import atexit

class CloudpickleWrapper(object):
    def __init__(self, var):
        """
        Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
        :param var: (Any) the variable you wish to wrap for pickling with cloudpickle
        """
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        self.var = cloudpickle.loads(obs)

def worker(child_connection, pickledFunc):
    envBuilder = pickle.loads(pickledFunc)
    env = envBuilder()
    # print("######################## Worker pid = "+str(os.getpid()))
    while True:
        try:
            cmd, data = child_connection.recv()

            if cmd == "submitAction":
                child_connection.send(env.submitAction(*data))
            elif cmd == "checkEpisodeEnded":
                child_connection.send(env.checkEpisodeEnded(*data))
            elif cmd == "computeReward":
                child_connection.send(env.computeReward(*data))
            elif cmd == "getObservation":
                child_connection.send(env.getObservation(*data))
            elif cmd == "getState":
                child_connection.send(env.getState())
            elif cmd == "getCameraToRenderName":
                child_connection.send(env.getCameraToRenderName())
            elif cmd == "initializeEpisode":
                child_connection.send(env.initializeEpisode())
            elif cmd == "performStep":
                child_connection.send(env.performStep())
            elif cmd == "performReset":
                child_connection.send(env.performReset())
            elif cmd == "getUiRendering":
                child_connection.send(env.getUiRendering())
            elif cmd == "getInfo":
                child_connection.send(env.getInfo(*data))
            elif cmd == "getMaxStepsPerEpisode":
                child_connection.send(env.getMaxStepsPerEpisode())
            elif cmd == "setGoalInState":
                child_connection.send(env.setGoalInState())
            elif cmd == "buildSimulation":
                child_connection.send(env.buildSimulation(*data))
            elif cmd == "close":
                child_connection.send(env.close())
                child_connection.close()
                break
            elif cmd == "getSimTimeFromEpStart":
                child_connection.send(env.getSimTimeFromEpStart())
            elif cmd == 'get_spaces':
                child_connection.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                child_connection.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                child_connection.send(getattr(env, data))
            elif cmd == 'set_attr':
                child_connection.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError("`{}` is not implemented in the worker".format(cmd))
        except EOFError:
            break

class SubProcGazeboEnvWrapper(lr_gym.envs.BaseEnv.BaseEnv):

    action_space = None
    observation_space = None
    metadata = None # e.g. {'render.modes': ['rgb_array']}

    def __init__(self,
                 constructEnvFunction : Callable,
                 start_method : str = None):
        """

        """

        print("######################## Main pid = "+str(os.getpid()))


        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)
        parent_conn, child_conn = Pipe()
        self._parentConnection = parent_conn
        pickledFunc = cloudpickle.dumps(constructEnvFunction)
        self._process = ctx.Process(target=worker, args=(child_conn, pickledFunc), daemon=True)
        self._process.start()
        child_conn.close()

        self._parentConnection.send(('get_spaces', None))
        self.observation_space, self.action_space = self._parentConnection.recv()
        self._parentConnection.send(('get_attr', "metadata"))
        self.metadata = self._parentConnection.recv()

        atexit.register(self.close)

    def submitAction(self, action) -> None:
        self._parentConnection.send(('submitAction', (action,)))
        return self._parentConnection.recv()

    def checkEpisodeEnded(self, previousState, state) -> bool:
        self._parentConnection.send(('checkEpisodeEnded', (previousState, state)))
        return self._parentConnection.recv()

    def computeReward(self, previousState, state, action) -> float:
        self._parentConnection.send(('computeReward', (previousState, state, action)))
        return self._parentConnection.recv()

    def getObservation(self, state) -> np.ndarray:
        self._parentConnection.send(('getObservation', (state,)))
        return self._parentConnection.recv()

    def getState(self) -> Sequence:
        self._parentConnection.send(('getState', None))
        return self._parentConnection.recv()

    def getCameraToRenderName(self) -> str:
        self._parentConnection.send(('getCameraToRenderName', None))
        return self._parentConnection.recv()

    def initializeEpisode(self) -> None:
        self._parentConnection.send(('initializeEpisode', None))
        return self._parentConnection.recv()

    def performStep(self) -> None:
        self._parentConnection.send(('performStep', None))
        return self._parentConnection.recv()

    def performReset(self) -> None:
        self._parentConnection.send(('performReset', None))
        return self._parentConnection.recv()

    def getUiRendering(self) -> Tuple[np.ndarray, float]:
        self._parentConnection.send(('getUiRendering', None))
        return self._parentConnection.recv()

    def getInfo(self,state=None) -> Dict[Any,Any]:
        self._parentConnection.send(('getInfo', (state,)))
        return self._parentConnection.recv()

    def getMaxStepsPerEpisode(self):
        self._parentConnection.send(('getMaxStepsPerEpisode', None))
        return self._parentConnection.recv()

    def setGoalInState(self, state, goal):
        self._parentConnection.send(('setGoalInState', (state, goal)))
        return self._parentConnection.recv()

    def buildSimulation(self, backend : str = "gazebo"):
        self._parentConnection.send(('buildSimulation', (backend,)))
        return self._parentConnection.recv()

    def close(self):
        self._parentConnection.send(('close', None))
        self._parentConnection.recv()
        self._parentConnection.close()

        self._process.join(30)
        if self._process.is_alive():
            os.kill(self._process.pid, signal.SIGINT)
            self._process.join(30)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(30)
                if self._process.is_alive():
                    self._process.kill()

    def getSimTimeFromEpStart(self):
        self._parentConnection.send(('getSimTimeFromEpStart', None))
        return self._parentConnection.recv()
