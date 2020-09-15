
#include <GravityCompensatedEffortController.hpp>
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

  bool GravityCompensatedEffortController::init(hardware_interface::EffortJointInterface* hw, ros::NodeHandle &n)
  {
    // List of controlled joints
    std::string joint_chain_param_name = "joint_chain";
    if(!n.getParam(joint_chain_param_name, joint_names))
    {
      ROS_ERROR_STREAM("Failed to getParam '" << joint_chain_param_name << "' (namespace: " << n.getNamespace() << ").");
      return false;
    }
    n_joints_ = joint_names.size();

    if(n_joints_ == 0)
    {
      ROS_ERROR_STREAM("List of joint names is empty.");
      return false;
    }
    for(unsigned int i=0; i<n_joints_; i++)
    {
      try
      {
        joints_.push_back(hw->getHandle(joint_names[i]));
      }
      catch (const hardware_interface::HardwareInterfaceException& e)
      {
        ROS_ERROR_STREAM("Exception thrown: " << e.what());
        return false;
      }
    }

    command_buffer.writeFromNonRT(std::vector<double>(n_joints_, 0.0));
    gravityCompensationTorques.writeFromNonRT(KDL::JntArray(n_joints_));
    lastTimeReceivedJointStates.writeFromNonRT(std::chrono::time_point<std::chrono::steady_clock>());//initializes to epoch time to epoch is zero, (1970)



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

    KDL::Segment firstSegment;
    bool r = getSegmentForJoint(joint_names[0], tree, firstSegment);
    if(!r)
    {
      ROS_ERROR_STREAM("Could not get segment (=link) for joint "<<joint_names[0]);
      return false;
    }
    KDL::Segment rootSegment;
    r = getSegmentParent(firstSegment, tree, rootSegment);
    if(!r)
    {
      ROS_ERROR_STREAM("Could not get parent of segment (=link) "<<firstSegment.getName()<<". First segment must have a parent to create a KDL chain.");
      return false;
    }

    KDL::Segment tipSegment;
    r = getSegmentForJoint(joint_names.back(), tree, tipSegment);
    if(!r)
    {
      ROS_ERROR_STREAM("Could not get segment (=link) for joint "<<joint_names.back());
      return false;
    }
    std::string rootLinkName = rootSegment.getName();
    std::string tipLinkName = tipSegment.getName();
    ROS_INFO_STREAM("RootLinkName = "<<rootLinkName<<"   tipLinkName = "<<tipLinkName);

    r = tree.getChain(rootLinkName,tipLinkName,robotChain);
    if(!r)
    {
      ROS_ERROR_STREAM("KDL::Tree::getChain failed. Cannot get chain from link "<<rootLinkName<<" to "<<tipLinkName);
      return false;
    }

    ROS_INFO_STREAM("Built chain with segments:");
    for(KDL::Segment& seg : robotChain.segments)
      ROS_INFO_STREAM(" - \""<<seg.getName()<<"\" (joint = \""<<seg.getJoint().getName()<<"\" of type "<<seg.getJoint().getTypeName()<<")");


    for(unsigned int i=0; i<n_joints_; i++)
    {
      if(robotChain.getSegment(i).getJoint().getName() != joint_names.at(i))
      {
        ROS_ERROR_STREAM("Joints specified in "<<joint_chain_param_name<<" do not correspond to a chain starting from joint "<<robotChain.getSegment(0).getJoint().getName()<<
                         " ("<<robotChain.getSegment(i).getJoint().getName() <<" != "<< joint_names.at(i)<<", i = "<<i<<")");
        return false;
      }
    }

    KDL::Vector gravityVector(0,0,-9.80665);
    chainDynParam = std::make_shared<KDL::ChainDynParam>(robotChain, gravityVector);





    sub_command_ = n.subscribe<std_msgs::Float64MultiArray>("command", 1, &GravityCompensatedEffortController::commandCB, this);
    jointStatesSub = n.subscribe("/joint_states", 1, &GravityCompensatedEffortController::jointStatesCallback, this);
    return true;
  }

  void GravityCompensatedEffortController::starting(const ros::Time& time)
  {
    // Start controller with 0.0 efforts
    command_buffer.readFromRT()->assign(n_joints_, 0.0);
  }


  void GravityCompensatedEffortController::commandCB(const std_msgs::Float64MultiArrayConstPtr& msg)
  {
    if(msg->data.size()!=n_joints_)
    {
      ROS_ERROR_STREAM("Dimension of command (" << msg->data.size() << ") does not match number of joints (" << n_joints_ << ")! Not executing!");
      return;
    }
    command_buffer.writeFromNonRT(msg->data);
  }

  void GravityCompensatedEffortController::jointStatesCallback(const sensor_msgs::JointStateConstPtr& msg)
  {
    KDL::JntArray joint_positions(joint_names.size());
    ROS_INFO("Received joint states");

    int i=0;
    for(std::string jointName : joint_names)
    {
      auto jointIt = std::find(msg->name.begin(),msg->name.end(),jointName);
      if(jointIt == msg->name.end())
      {
        ROS_WARN_STREAM("Joint "<<jointName<<" not found in received JointState, skipping message");
        return;
      }
      int idx = jointIt-msg->name.begin();
      ROS_INFO_STREAM("found joint "<<jointName<<" at idx "<<idx);
      joint_positions(i++) = msg->position.at(idx);
    }


    KDL::JntArray gravityCompensationTorquesArr(joint_names.size());
    chainDynParam->JntToGravity(joint_positions,gravityCompensationTorquesArr);
    ROS_INFO_STREAM("Computed gravity compensation");

    gravityCompensationTorques.writeFromNonRT(gravityCompensationTorquesArr);
  }


  void GravityCompensatedEffortController::update(const ros::Time& /*time*/, const ros::Duration& /*period*/)
  {
    std::vector<double> command = *command_buffer.readFromRT();
    KDL::JntArray gravityCompensationTorquesArr = *(gravityCompensationTorques.readFromRT());
    for(unsigned int i=0; i<n_joints_; i++)
    {
      joints_[i].setCommand(command[i] + gravityCompensationTorquesArr(i));
    }
  }
}

PLUGINLIB_EXPORT_CLASS(gazebo_gym_utils::GravityCompensatedEffortController, controller_interface::ControllerBase)
