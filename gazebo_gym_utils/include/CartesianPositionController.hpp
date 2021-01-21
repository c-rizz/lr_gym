
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
    KDL::JntArray computeIk(std::vector<double> requestedCartesianPose);
    std::vector<double> getCurrentJointPose();

    std::vector< hardware_interface::JointHandle > joints_;
    
    KDL::Chain robotChain;
    std::vector<std::string> notFixedJointsNames;
    std::shared_ptr<KDL::ChainDynParam> chainDynParam;
    KDL::JntArray joint_limits_low;
    KDL::JntArray joint_limits_high;


    realtime_tools::RealtimeBuffer<std::vector<double> > command_buffer;

    ros::Subscriber sub_command_;
    void commandCB(const std_msgs::Float64MultiArrayConstPtr& msg);
  };

}
