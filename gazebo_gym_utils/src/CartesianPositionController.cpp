
#include "../include/CartesianPositionController.hpp"
#include "../include/KdlHelper.hpp"
#include <pluginlib/class_list_macros.h>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_pinv_givens.hpp>

namespace gazebo_gym_utils
{

  CartesianPositionController::CartesianPositionController()
  {

  }





  CartesianPositionController::~CartesianPositionController()
  {
    sub_command_.shutdown();
  }








  bool CartesianPositionController::init(hardware_interface::PositionJointInterface* hw, ros::NodeHandle &n)
  {
    // List of controlled joints
    std::string joint_chain_param_name = "joint_chain";
    std::string robot_description_param_name = "robot_description";
    std::vector<std::string> joint_names;
    if(!n.getParam(joint_chain_param_name, joint_names))
    {
      ROS_ERROR_STREAM("Failed to getParam '" << joint_chain_param_name << "' (namespace: " << n.getNamespace() << ").");
      return false;
    }

    if(joint_names.size() == 0)
    {
      ROS_ERROR_STREAM("List of joint names is empty.");
      return false;
    }


    command_buffer.writeFromNonRT(getCurrentJointPose());

    std::string joint_names_str = "";
    for(std::string jn : joint_names)
      joint_names_str+=jn +", ";
    ROS_INFO_STREAM("Got joint_names = ["<<joint_names_str<<"]");

    try
    {
      robotChain = KdlHelper::getChainFromJoints(joint_names, robot_description_param_name);
    }
    catch(const std::runtime_error& e)
    {
      ROS_ERROR_STREAM("Failed to get chain for the specified joints: "<<e.what());
      return false;
    }


    ROS_INFO_STREAM("Built chain with segments:");
    for(KDL::Segment& seg : robotChain.segments)
    {
      if(seg.getJoint().getType() != KDL::Joint::JointType::None)
      {
        ROS_INFO_STREAM("Joint "<<seg.getJoint().getName()<<" is not a fixed joint, type = "<<seg.getJoint().getTypeName());
        notFixedJointsNames.push_back(seg.getJoint().getName());
      }
    }

    for(std::string jn : notFixedJointsNames)
    {
      try
      {
        joints_.push_back(hw->getHandle(jn));
      }
      catch (const hardware_interface::HardwareInterfaceException& e)
      {
        ROS_ERROR_STREAM("Exception thrown: " << e.what());
        return false;
      }
    }


    std::vector<std::pair<double, double>> jointLimits = KdlHelper::getJointLimits(robot_description_param_name, notFixedJointsNames);

    joint_limits_low = KDL::JntArray(notFixedJointsNames.size());
    joint_limits_high = KDL::JntArray(notFixedJointsNames.size());
    for(unsigned int i = 0 ; i<jointLimits.size(); i++)
    {
      joint_limits_low(i) = jointLimits[i].first;
      joint_limits_high(i) = jointLimits[i].second;
    }

    sub_command_ = n.subscribe<std_msgs::Float64MultiArray>("command", 1, &CartesianPositionController::commandCB, this);
    return true;
  }




  std::vector<double> CartesianPositionController::getCurrentJointPose()
  {
    std::vector<double> joint_position = std::vector<double>(notFixedJointsNames.size());
    int i=0;
    for(hardware_interface::JointHandle jh : joints_)
      joint_position[i++] = jh.getPosition();
    return joint_position;
  }






  void CartesianPositionController::starting(const ros::Time& time)
  {
    // Start controller with curent pose
    *(command_buffer.readFromRT()) = getCurrentJointPose();
  }










  void CartesianPositionController::commandCB(const std_msgs::Float64MultiArrayConstPtr& msg)
  {
    if(msg->data.size()!=7)
    {
      ROS_ERROR_STREAM("Dimension of command (" << msg->data.size() << ") does not match size. Should be (pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w). Ignoring command.");
      return;
    }
    command_buffer.writeFromNonRT(msg->data);
  }





  KDL::JntArray CartesianPositionController::computeIk(std::vector<double> requestedCartesianPose)
  {
    KDL::ChainFkSolverPos_recursive fkpossolver(robotChain);
    KDL::ChainIkSolverVel_pinv_givens ikvelsolver(robotChain);
    KDL::ChainIkSolverPos_NR_JL inv_pos_solver(robotChain,
                                               joint_limits_low,joint_limits_high,
                                               fkpossolver,ikvelsolver);

    KDL::Frame requestedCartesianPose_frame = KDL::Frame(KDL::Rotation::Quaternion( requestedCartesianPose[3],
                                                                                    requestedCartesianPose[4],
                                                                                    requestedCartesianPose[5],
                                                                                    requestedCartesianPose[6]),
                                                         KDL::Vector(requestedCartesianPose[0],
                                                                     requestedCartesianPose[1],
                                                                     requestedCartesianPose[2]));


    KDL::JntArray q_init(robotChain.getNrOfJoints());
    for(unsigned int i=0; i < robotChain.getNrOfJoints(); i++ )
        q_init(i) = (joint_limits_high(i)-joint_limits_low(i))/2;

    KDL::JntArray jointPose(robotChain.getNrOfJoints());

    int status = inv_pos_solver.CartToJnt(q_init,
                                          requestedCartesianPose_frame,
                                          jointPose);

    if(status<0)
    {
      std::string cart_pose_str = "[";
      for(double v : requestedCartesianPose)
        cart_pose_str += std::to_string(v);
      cart_pose_str += "]";
      throw std::runtime_error("Failed to compute IK for cartesian pose "+cart_pose_str+". status = "+std::to_string(status));
    }
    return jointPose;
  }



  void CartesianPositionController::update(const ros::Time& /*time*/, const ros::Duration& /*period*/)
  {
    std::vector<double> requestedCartesianPose = *command_buffer.readFromRT();
    
    KDL::JntArray jointPose;
    try
    {
      jointPose = computeIk(requestedCartesianPose);
    }
    catch(const std::runtime_error& e)
    {
      std::string err = std::string("Failed to compute inverse kinematics: ")+e.what();
      ROS_ERROR_STREAM(err);
      return;
    }


    for(unsigned int i=0; i<notFixedJointsNames.size(); i++)
      joints_[i].setCommand(jointPose(i));
  }
}

PLUGINLIB_EXPORT_CLASS(gazebo_gym_utils::CartesianPositionController, controller_interface::ControllerBase)
