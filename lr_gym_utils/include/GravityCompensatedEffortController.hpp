
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

namespace lr_gym_utils
{


  /**
   * \brief Controls a set of joint with effort control, adding gravity compensation torques
   *
   * \section ROS interface
   *
   * \param joints Names of the joints to control.
   *
   * Subscribes to:
   * - \b command (std_msgs::Float64MultiArray) : The joint commands to apply.
   */
  class GravityCompensatedEffortController: public controller_interface::Controller<hardware_interface::EffortJointInterface>
  {
  public:
    GravityCompensatedEffortController();

    ~GravityCompensatedEffortController();

    bool init(hardware_interface::EffortJointInterface* hw, ros::NodeHandle &n);

    void starting(const ros::Time& time);
    void update(const ros::Time& /*time*/, const ros::Duration& /*period*/);

private:
    KDL::JntArray computeGravityCompensation();

    std::vector< hardware_interface::JointHandle > joints_;
    double gravity_acceleration;

    KDL::Chain robotChain;
    std::vector<std::string> notFixedJointsNames;
    std::shared_ptr<KDL::ChainDynParam> chainDynParam;

    realtime_tools::RealtimeBuffer<std::vector<double> > command_buffer;

    ros::Subscriber sub_command_;
    void commandCB(const std_msgs::Float64MultiArrayConstPtr& msg);
  };

}
