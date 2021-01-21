
#include "../include/GravityCompensatedEffortController.hpp"
#include "../include/KdlHelper.hpp"
#include <pluginlib/class_list_macros.h>


namespace gazebo_gym_utils
{

  GravityCompensatedEffortController::GravityCompensatedEffortController()
  {

  }





  GravityCompensatedEffortController::~GravityCompensatedEffortController()
  {
    sub_command_.shutdown();
  }








  bool GravityCompensatedEffortController::init(hardware_interface::EffortJointInterface* hw, ros::NodeHandle &n)
  {
    // List of controlled joints
    std::string joint_chain_param_name = "joint_chain";
    std::vector<std::string> joint_names;
    if(!n.getParam(joint_chain_param_name, joint_names))
    {
      ROS_ERROR_STREAM("Failed to getParam '" << joint_chain_param_name << "' (namespace: " << n.getNamespace() << ").");
      return false;
    }

    std::string g_param_name = "gravity_acceleration";
    if(!n.getParam(g_param_name, gravity_acceleration))
    {
      ROS_ERROR_STREAM("Failed to getParam '" << g_param_name << "' (namespace: " << n.getNamespace() << ").");
      return false;
    }

    if(joint_names.size() == 0)
    {
      ROS_ERROR_STREAM("List of joint names is empty.");
      return false;
    }

    command_buffer.writeFromNonRT(std::vector<double>(joint_names.size(), 0.0));

    std::string joint_names_str = "";
    for(std::string jn : joint_names)
      joint_names_str+=jn +", ";
    ROS_INFO_STREAM("Got joint_names = ["<<joint_names_str<<"]");

    try
    {
      robotChain = KdlHelper::getChainFromJoints(joint_names, "robot_description");
    }
    catch(const std::runtime_error& e)
    {
      ROS_ERROR_STREAM("Failed to get chain for the specified joints: "<<e.what());
      return false;
    }

    KDL::Vector gravityVector(0,0,-gravity_acceleration);
    chainDynParam = std::make_shared<KDL::ChainDynParam>(robotChain, gravityVector);

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

    sub_command_ = n.subscribe<std_msgs::Float64MultiArray>("command", 1, &GravityCompensatedEffortController::commandCB, this);
    return true;
  }











  void GravityCompensatedEffortController::starting(const ros::Time& time)
  {
    // Start controller with 0.0 efforts
    command_buffer.readFromRT()->assign(notFixedJointsNames.size(), 0.0);
  }










  void GravityCompensatedEffortController::commandCB(const std_msgs::Float64MultiArrayConstPtr& msg)
  {
    if(msg->data.size()!=notFixedJointsNames.size())
    {
      ROS_ERROR_STREAM("Dimension of command (" << msg->data.size() << ") does not match number of joints (" << notFixedJointsNames.size() << ")! Not executing!");
      return;
    }
    command_buffer.writeFromNonRT(msg->data);
  }











  KDL::JntArray GravityCompensatedEffortController::computeGravityCompensation()
  {
    KDL::JntArray joint_positions(joints_.size());
    //ROS_INFO("Received joint states");

    int i=0;
    for(hardware_interface::JointHandle jh : joints_)
      joint_positions(i++) = jh.getPosition();


    KDL::JntArray gravityCompensationTorquesArr(notFixedJointsNames.size());
    chainDynParam->JntToGravity(joint_positions,gravityCompensationTorquesArr);
    std::string torquesStr = "[";
    for(unsigned int i=0; i<notFixedJointsNames.size()-1; i++)
      torquesStr += std::to_string(gravityCompensationTorquesArr(i))+",\t";
    torquesStr += std::to_string(gravityCompensationTorquesArr(notFixedJointsNames.size()-1))+"]";
    //ROS_INFO_STREAM("Computed gravity compensation: "<<torquesStr);

    return gravityCompensationTorquesArr;
  }











  void GravityCompensatedEffortController::update(const ros::Time& /*time*/, const ros::Duration& /*period*/)
  {
    std::vector<double> command = *command_buffer.readFromRT();
    KDL::JntArray gravityCompensationTorquesArr = computeGravityCompensation();
    std::vector<double> totCommand;
    for(unsigned int i = 0; i<command.size(); i++)
      totCommand.push_back(command.at(i)+ gravityCompensationTorquesArr(i));


    std::string torquesStr = "[";
    for(unsigned int i=0; i<notFixedJointsNames.size()-1; i++)
      torquesStr += std::to_string(totCommand.at(i))+",\t";
    torquesStr += std::to_string(totCommand.at(notFixedJointsNames.size()-1))+"]";
    //ROS_INFO_STREAM("Commanding torques:: "<<torquesStr);


    for(unsigned int i=0; i<notFixedJointsNames.size(); i++)
    {
      joints_[i].setCommand(totCommand[i]);
    }
  }
}

PLUGINLIB_EXPORT_CLASS(gazebo_gym_utils::GravityCompensatedEffortController, controller_interface::ControllerBase)
