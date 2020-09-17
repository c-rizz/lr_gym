
#include "../include/GravityCompensatedEffortController.hpp"
#include <pluginlib/class_list_macros.h>
#include <urdf/model.h>


namespace gazebo_gym_utils
{

  GravityCompensatedEffortController::GravityCompensatedEffortController()
  {

  }

  GravityCompensatedEffortController::~GravityCompensatedEffortController()
  {
    sub_command_.shutdown();
  }

  bool GravityCompensatedEffortController::getSegmentForJoint(std::string jointName, KDL::Tree tree, KDL::Segment& segment)
  {
    ROS_INFO_STREAM("Getting segment for joint "<<jointName);
    std::vector<KDL::Segment> candidateSegments;
    KDL::SegmentMap allSegments = tree.getSegments();
    for(auto it : allSegments)
    {
      ROS_INFO_STREAM("Segment "<<it.second.segment.getName()<<" has joint "<<it.second.segment.getJoint().getName());
      if(it.second.segment.getJoint().getName() == jointName)
        candidateSegments.push_back(it.second.segment);

    }
    if(candidateSegments.size()>1)
    {
      ROS_ERROR_STREAM("There are multiple links for joint "<<jointName<<". Cannot get link for joint.");
      return false;
    }
    if(candidateSegments.size()<1)
    {
      ROS_ERROR_STREAM("No links for joint "<<jointName<<". Cannot get link for joint.");
      return false;
    }
    segment = candidateSegments[0];
    return true;
  }


  bool GravityCompensatedEffortController::getSegmentParent(KDL::Segment& segment, KDL::Tree tree, KDL::Segment& parent)
  {
    KDL::SegmentMap allSegments = tree.getSegments();
    for(auto it : allSegments)
    {
      //ROS_INFO_STREAM("Segment "<<it.second.segment.getName()<<" has joint "<<it.second.segment.getJoint().getName());
      if(it.second.segment.getName() == segment.getName())
        parent = it.second.parent->second.segment;
    }
    return true;
  }

  KDL::Chain GravityCompensatedEffortController::getChainFromJoints(const KDL::Tree& tree, std::vector<std::string> jointNames)
  {
    KDL::Segment firstSegment;
    bool r = getSegmentForJoint(jointNames[0], tree, firstSegment);
    if(!r)
      throw std::runtime_error("Could not get segment (=link) for joint "+jointNames[0]);

    KDL::Segment rootSegment;
    r = getSegmentParent(firstSegment, tree, rootSegment);
    if(!r)
      throw std::runtime_error("Could not get parent of segment (=link) "+firstSegment.getName()+". First segment must have a parent to create a KDL chain.");

    KDL::Segment tipSegment;
    r = getSegmentForJoint(jointNames.back(), tree, tipSegment);
    if(!r)
      throw std::runtime_error("Could not get segment (=link) for joint "+jointNames.back());

    std::string rootLinkName = rootSegment.getName();
    std::string tipLinkName = tipSegment.getName();
    ROS_INFO_STREAM("RootLinkName = "<<rootLinkName<<"   tipLinkName = "<<tipLinkName);

    KDL::Chain chain;
    r = tree.getChain(rootLinkName,tipLinkName,chain);
    if(!r)
      throw std::runtime_error("KDL::Tree::getChain failed. Cannot get chain from link "+rootLinkName+" to "+tipLinkName);

    ROS_INFO_STREAM("Built chain with segments:");
    for(KDL::Segment& seg : chain.segments)
      ROS_INFO_STREAM(" - \""<<seg.getName()<<"\" (joint = \""<<seg.getJoint().getName()<<"\" of type "<<seg.getJoint().getTypeName()<<")");


    for(unsigned int i=0; i<jointNames.size(); i++)
    {
      if(chain.getSegment(i).getJoint().getName() != jointNames.at(i))
      {
        throw std::runtime_error( "Specified joints do not correspond to a chain starting from joint "+
                                  chain.getSegment(0).getJoint().getName()+
                                  " ("+chain.getSegment(i).getJoint().getName() +" != "+ jointNames.at(i)+", i = "+std::to_string(i)+")");
      }
    }

    return chain;
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
      ROS_ERROR_STREAM("Failed to getParam '" << gravity_acceleration << "' (namespace: " << n.getNamespace() << ").");
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


    // gets the location of the robot description on the parameter server
    urdf::Model model;
    if (!model.initParam("robot_description"))
    {
      ROS_ERROR("Failed to get robot_description");
      return false;
    }

    KDL::Tree tree;
    if (!kdl_parser::treeFromUrdfModel(model, tree))
    {
      ROS_ERROR("Failed to extract kdl tree from xml robot description");
      return false;
    }

    try
    {
      robotChain = getChainFromJoints(tree, joint_names);
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
