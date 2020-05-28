#include <mutex>
#include <vector>
#include <ros/ros.h>

#include "gazebo_gym_env_plugin/JointEffortRequest.h"
#include "JointEffortControl.hpp"


JointEffortControl::JointEffortControl(gazebo::physics::WorldPtr world)
{
  this->world = world;
  applyJointEffortsCallback = gazebo::event::Events::ConnectWorldUpdateBegin(boost::bind(&JointEffortControl::applyRequestedJointEfforts,this));
}
/**
 * Set the effort to be applied on a joint in the next timestep
 * @param  jointId Identifier for the joint
 * @param  effort  Effort to be applied (force or torque depending on the joint type)
 * @return         0 if successfult, negative otherwise
 */
int JointEffortControl::setJointEffort(const gazebo_gym_env_plugin::JointId& jointId, double effort)
{
  ROS_DEBUG_STREAM("Applying effort="<<effort<<" to joint "<<jointId.model_name<<"."<<jointId.joint_name);
  gazebo::physics::ModelPtr model = world->ModelByName(jointId.model_name);
  if (!model)
    return -1;
  gazebo::physics::JointPtr joint = model->GetJoint(jointId.joint_name);
  if (!joint)
    return -2;

  joint->SetForce(0,effort); //TODO: do something for joints with more than 1 DOF
  return 0;
}

/**
 * Applies the efforts requested by requestJointEffort()
 * @return 0 if successfult, negative if any joint effort request failed
 */
int JointEffortControl::applyRequestedJointEfforts()
{
  std::unique_lock<std::timed_mutex> lk(requestedJointEfforts_mutex,std::chrono::seconds(5));
  if(!lk)
  {
    ROS_ERROR_STREAM("Failed to acquire mutex in "<<__func__<<", aborting");
    return -1;
  }
  //ROS_INFO("Appling efforts");
  int ret = 0;
  for(const gazebo_gym_env_plugin::JointEffortRequest& jer : requestedJointEfforts)
  {
    int r = setJointEffort(jer.joint_id,jer.effort);
    if(r<0)
    {
      ROS_ERROR_STREAM("Failed to apply effort to joint "<<jer.joint_id.model_name<<"."<<jer.joint_id.joint_name);
      ret = -2;
    }
  }
  return ret;
}

/**
 * Request a joint effort to be applied on the next timestep
 * @param  request Joint effort request
 * @return         0 if successful
 */
int JointEffortControl::requestJointEffort(const gazebo_gym_env_plugin::JointEffortRequest& request)
{
  std::unique_lock<std::timed_mutex> lk(requestedJointEfforts_mutex,std::chrono::seconds(5));
  if(!lk)
  {
    ROS_ERROR_STREAM("Failed to acquire mutex in "<<__func__<<", aborting");
    return -1;
  }
  requestedJointEfforts.push_back(request);
  return 0;
}

/**
 * Clear requests made via requestJointEffort()
 * @return         0 if successful
 */
int JointEffortControl::clearRequestedJointEfforts()
{
  std::unique_lock<std::timed_mutex> lk(requestedJointEfforts_mutex,std::chrono::seconds(5));
  if(!lk)
  {
    ROS_ERROR_STREAM("Failed to acquire mutex in "<<__func__<<", aborting");
    return -1;
  }
  requestedJointEfforts.clear();
  return 0;
}
