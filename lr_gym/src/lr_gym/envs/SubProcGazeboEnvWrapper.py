from collections.abc import Callable
import multiprocessing
from multiprocessing import Pipe
import cloudpickle
import pickle
from typing import Sequence, Dict, Any, Tuple
import lr_gym
import numpy as np
from lr_gym.envs.BaseEnv import BaseEnv

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
            funcname, args, kwargs = child_connection.recv()


            if funcname == 'get_spaces':
                child_connection.send((env.observation_space, env.action_space))
            elif funcname == 'env_method':
                method = getattr(env, args[0])
                child_connection.send(method(*args[1], **args[2]))
            elif funcname == 'get_attr':
                child_connection.send(getattr(env, *args))
            elif funcname == 'set_attr':
                child_connection.send(setattr(env, args[0], args[1]))
            else:
                ret = getattr(env, funcname)(*args, **kwargs)
                child_connection.send(ret)
                if funcname == "close":
                    child_connection.close()
                # raise NotImplementedError("`{}` is not implemented in the worker".format(funcname))
        except EOFError:
            break

class SubProcGazeboEnvWrapper():

    action_space = None
    observation_space = None
    metadata = None # e.g. {'render.modes': ['rgb_array']}

    def __init__(self,
                 constructEnvFunction : Callable,
                 start_method : str = None):
        """

        """

        # print("######################## Main pid = "+str(os.getpid()))


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

        self.observation_space, self.action_space = self.__call_in_subproc('get_spaces')
        self.metadata = self.__call_in_subproc('get_attr', ("metadata",))

        atexit.register(self.close)

    def __call_in_subproc(self, name, args = (), kwargs = {}):
        self._parentConnection.send((name, tuple(args), kwargs))
        return self._parentConnection.recv()
    
    def __getattr__(self, name):
        '''
        For any attribute not in the wrapper try to call from self.env
        '''
        return lambda *args, **kwargs : self.__call_in_subproc(name, args, kwargs)


    def close(self):
        self.__call_in_subproc('close')
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

