#ifndef JOINT_EFFORT_CONTROL_CRZZ_20200528
#define JOINT_EFFORT_CONTROL_CRZZ_20200528

#include <mutex>
#include <vector>
#include "gazebo/physics/physics.hh"

#include "gazebo_gym_env_plugin/JointEffortRequest.h"


class JointEffortControl
{
private:
  std::vector<gazebo_gym_env_plugin::JointEffortRequest> requestedJointEfforts;
  std::timed_mutex requestedJointEfforts_mutex;
  gazebo::physics::WorldPtr world;
  gazebo::event::ConnectionPtr applyJointEffortsCallback;

public:

  JointEffortControl(gazebo::physics::WorldPtr world);
  /**
   * Set the effort to be applied on a joint in the next timestep
   * @param  jointId Identifier for the joint
   * @param  effort  Effort to be applied (force or torque depending on the joint type)
   * @return         0 if successfult, negative otherwise
   */
  int setJointEffort(const gazebo_gym_env_plugin::JointId& jointId, double effort);

  /**
   * Applies the efforts requested by requestJointEffort()
   * @return 0 if successfult, negative if any joint effort request failed
   */
  int applyRequestedJointEfforts();

  /**
   * Request a joint effort to be applied on the next timestep
   * @param  request Joint effort request
   * @return         0 if successful
   */
  int requestJointEffort(const gazebo_gym_env_plugin::JointEffortRequest& request);
  /**
   * Clear requests made via requestJointEffort()
   * @return         0 if successful
   */
  int clearRequestedJointEfforts();
};

#endif
