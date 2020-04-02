#!/usr/bin/env python3
import traceback
import typing

import rospy
from std_srvs.srv import Empty

class GazeboController():
    """This class allows to control the execution of the Gazebo simulation
    """

    def __init__(self, usePersistentConnections : bool = True):
        """Initializes the Gazebo controller

        Parameters
        ----------
        usePersistentConnections : bool
            Controls wheter to use persistent connections for the gazebo services.
            Should be fine as long as there are no connection problems and gazebo does not restart

        Raises
        -------
        ROSException
            If it fails to find the gazebo services

        """

        self._pauseGazeboServiceName = "/gazebo/pause_physics"
        self._unpauseGazeboServiceName = "/gazebo/unpause_physics"
        self._resetGazeboWorldService = "/gazebo/reset_world"
        self._setGazeboPhysics = "/gazebo/set_physics_properties"

        timeout_secs = 30.0
        try:
            rospy.wait_for_service(self._pauseGazeboServiceName,timeout_secs)
            rospy.wait_for_service(self._unpauseGazeboServiceName,timeout_secs)
            rospy.wait_for_service(self._resetGazeboWorldService,timeout_secs)
            rospy.wait_for_service(self._setGazeboPhysics,timeout_secs)
        except rospy.ROSException as e:
            rospy.logfatal("Failed to wait for Gazebo services. Timeout was "+str(timeout_secs)+"s")
            raise
        except rospy.ROSInterruptException as e:
            rospy.logfatal("Interrupeted while waiting for Gazebo services.")
            raise

        self._pauseGazeboService = rospy.ServiceProxy(self._pauseGazeboServiceName, Empty, persistent=usePersistentConnections)
        self._unpauseGazeboService = rospy.ServiceProxy(self._unpauseGazeboServiceName, Empty, persistent=usePersistentConnections)
        self._resetGazeboWorldService = rospy.ServiceProxy(self._resetGazeboWorldService, Empty, persistent=usePersistentConnections)
        #self._setGazeboPhysics = rospy.ServiceProxy(self._setGazeboPhysics, SetPhysicsProperties, persistent=usePersistentConnections)

        self.pauseSimulation()
        self.resetWorld()

    def _callService(self,serviceProxy : rospy.ServiceProxy) -> bool:
        """Call the provided service. It retries in case of failure and handles exceptions. Returns false if the call failed

        Parameters
        ----------
        serviceProxy : rospy.ServiceProxy
            ServiceProxy for the service to be called

        Returns
        -------
        bool
            True if the service was called, false otherwise

        """
        done = False
        counter = 0
        maxRetry = 10
        while not done and not rospy.is_shutdown():
            if counter < maxRetry:
                try:
                    serviceProxy.call()
                    done = True
                except rospy.ServiceException as e:
                    rospy.logerr("Service "+serviceProxy.resolved_name+", call failed: "+traceback.format_exc(e))
                except rospy.ROSInterruptException as e:
                    rospy.logerr("Service "+serviceProxy.resolved_name+", call interrupted: "+traceback.format_exc(e))
                    counter+=maxRetry #don't retry
                except rospy.ROSSerializationException as e:
                    rospy.logerr("Service "+serviceProxy.resolved_name+", call failed to serialize: "+traceback.format_exc(e))
                counter += 1
            else:
                rospy.logerr("Failed to pause gazebo simulation")
                break
        return done

    def pauseSimulation(self) -> bool:
        """Pauses the simulation

        Returns
        -------
        bool
            True if the simulation was paused, false in case of failure

        """
        return self._callService(self._pauseGazeboService)

    def unpauseSimulation(self) -> bool:
        """Unpauses the simulation

        Returns
        -------
        bool
            True if the simulation was paused, false in case of failure

        """
        return self._callService(self._unpauseGazeboService)

    def resetWorld(self) -> bool:
        """Resets the world to its initial state

        Returns
        -------
        bool
            True if the simulation was paused, false in case of failure

        """
        return self._callService(self._resetGazeboWorldService)

    def unpauseSimulationFor(self, runTime_secs : float) -> None:
        """Runs the simulation for the specified time. It unpauses and the simulation, sleeps and then pauses it back.
        It may not be precise

        Parameters
        ----------
        runTime_secs : float
            Time to run the simulation for, in seconds

        Returns
        -------
        None


        Raises
        -------
        ExceptionName
            Why the exception is raised.

        """

        t0 = rospy.get_time()
        self.unpauseSimulation()
        t1 = rospy.get_time()
        rospy.sleep(runTime_secs)
        t2 = rospy.get_time()
        self.pauseSimulation()
        t3 = rospy.get_time()
        rospy.loginfo("Unpaused Gazebo for a duration between "+str(t2-t1)+"s and "+str(t3-t0)+"s")
