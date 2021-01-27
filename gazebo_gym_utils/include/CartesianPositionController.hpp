
#pragma once


#include <vector>
#include <string>

#include <ros/node_handle.h>
#include <hardware_interface/joint_command_interface.h>
#include <controller_interface/controller.h>
#include <std_msgs/Float64MultiArray.h>
#include <realtime_tools/realtime_buffer.h>
#include <sensor_msgs/JointState.h>

#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl_conversions/kdl_msg.h>
#include <kdl/chaindynparam.hpp>
#include <random>

namespace gazebo_gym_utils
{


  /**
   * \brief Controls a set of joint with cartesian position control
   *
   * \section ROS interface
   *
   * \param joints Names of the joints to control.
   *
   * Subscribes to:
   * - \b command (std_msgs::Float64MultiArray) : The joint commands to apply.
   */
  class CartesianPositionController: public controller_interface::Controller<hardware_interface::PositionJointInterface>
  {
  public:
    CartesianPositionController();

    ~CartesianPositionController();

    bool init(hardware_interface::PositionJointInterface* hw, ros::NodeHandle &n);

    void starting(const ros::Time& time);
    void update(const ros::Time& /*time*/, const ros::Duration& /*period*/);

private:

    int attempts;
    int iterations;
    double precision;

    std::vector< hardware_interface::JointHandle > joint_handles;

    KDL::Chain robotChain;
    std::vector<std::string> notFixedJointsNames;
    std::shared_ptr<KDL::ChainDynParam> chainDynParam;
    KDL::JntArray joint_limits_low;
    KDL::JntArray joint_limits_high;

    realtime_tools::RealtimeBuffer<std::vector<double> > command_buffer;
    std::vector<double> lastSuccessfulCommand;
    std::vector<double> lastJointPose;

    ros::Subscriber commandSubscriber;

    std::mt19937 randomEngine;

    void setJointPose(std::vector<double> jointPose);
    std::string poseToStr(std::vector<double> p);
    std::vector<double> computeFk(std::vector<double> jointPose);
    std::vector<double> computeIk(std::vector<double> requestedCartesianPose);
    std::vector<double> getCurrentJointPose();
    void commandCB(const std_msgs::Float64MultiArrayConstPtr& msg);
    KDL::JntArray randomJointPose();

  };

}
