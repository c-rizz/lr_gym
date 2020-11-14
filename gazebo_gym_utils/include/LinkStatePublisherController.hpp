
#pragma once

#include <controller_interface/controller.h>
#include <hardware_interface/joint_state_interface.h>
#include <memory>
#include <realtime_tools/realtime_publisher.h>
#include <sensor_msgs/JointState.h>
#include <gazebo_gym_utils/LinkStates.h>

#include "../include/ForwardKinematicsComputer.hpp"

namespace gazebo_gym_utils
{

/**
 * \brief Controller that publishes the state of all links in a robot.
 *
 * This controller publishes the state of all links in the robot_description urdf to a
 * topic of type \c gazebo_gym_utils/LinkStates. The following is a basic configuration of the controller.
 *
 * \code
 * joint_state_controller:
 *   type: gazebo_gym_utils/LinkStatePublisherController
 *   publish_rate: 50
 * \endcode
 *
 * The controller extracts the robot structure from theurdf and gets the joint state from a
 * hardware_interface::JointStateInterface
 * It's possible to optionally specify a set of extra joints not contained in a
 * \c hardware_interface::JointStateInterface with custom (and static) default values. The following is an example
 *  configuration specifying extra joints.
 *
 * \code
 * joint_state_controller:
 *   type: gazebo_gym_utils/LinkStatePublisherController
 *   publish_rate: 50
 *   extra_joints:
 *     - name:     'extra1'
 *       position: 10.0
 *       velocity: 20.0
 *       effort:   30.0
 *
 *     - name:     'extra2'
 *       position: -10.0
 *
 *     - name:     'extra3'
 * \endcode
 *
 * An unspecified joint position, velocity or acceleration defaults to zero.
 */
class LinkStatePublisherController: public controller_interface::Controller<hardware_interface::JointStateInterface>
{
public:
  LinkStatePublisherController() : publish_rate_(0.0) {}

  virtual bool init(hardware_interface::JointStateInterface* hw,
                    ros::NodeHandle&                         root_nh,
                    ros::NodeHandle&                         controller_nh);
  virtual void starting(const ros::Time& time);
  virtual void update(const ros::Time& time, const ros::Duration& /*period*/);
  virtual void stopping(const ros::Time& /*time*/);

private:
  std::vector<hardware_interface::JointStateHandle> joint_state_;
  std::shared_ptr<realtime_tools::RealtimePublisher<gazebo_gym_utils::LinkStates> > realtime_pub_;
  ros::Time last_publish_time_;
  double publish_rate_;
  unsigned int num_hw_joints_; ///< Number of joints present in the JointStateInterface, excluding extra joints
  sensor_msgs::JointState joint_state_msg;
  std::shared_ptr<ForwardKinematicsComputer> forwardKinematicsComputer;


  void addExtraJoints(const ros::NodeHandle& nh, sensor_msgs::JointState& msg);
};

}
