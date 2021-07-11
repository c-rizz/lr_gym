#include <algorithm>
#include <cstddef>
#include <pluginlib/class_list_macros.hpp>

#include "../include/LinkStatePublisherController.hpp"
#include <lr_gym_utils/LinkStates.h>


namespace lr_gym_utils
{

  bool LinkStatePublisherController::init(hardware_interface::JointStateInterface* hw,
                                  ros::NodeHandle&                         root_nh,
                                  ros::NodeHandle&                         controller_nh)
  {
    ROS_INFO("Initializing LinkStatePublisherController");
    // get all joint names from the hardware interface
    const std::vector<std::string>& joint_names = hw->getNames();
    num_hw_joints_ = joint_names.size();

    std::string joint_names_str = "";
    for(std::string jn : joint_names)
      joint_names_str+=jn +", ";
    ROS_INFO_STREAM("LinkStatePublisherController: Got joint_names = ["<<joint_names_str<<"]");


    // get publishing period
    if (!controller_nh.getParam("publish_rate", publish_rate_)){
      ROS_ERROR("LinkStatePublisherController: Parameter 'publish_rate' not set");
      return false;
    }

    forwardKinematicsComputer = std::make_shared<lr_gym_utils::ForwardKinematicsComputer>();

    // realtime publisher
    realtime_pub_.reset(new realtime_tools::RealtimePublisher<lr_gym_utils::LinkStates>(root_nh, "link_states", 4));

    // get joint handles
    for (unsigned i=0; i<num_hw_joints_; i++)
      joint_state_.push_back(hw->getHandle(joint_names[i]));

    // Allocate joint state message
    for(unsigned i=0; i<num_hw_joints_; i++)
    {
      joint_state_msg.name.push_back(joint_names[i]);
      joint_state_msg.position.push_back(0.0);
      joint_state_msg.velocity.push_back(0.0);
      joint_state_msg.effort.push_back(0.0);
    }
    addExtraJoints(controller_nh, joint_state_msg);

    // Allocate link state message for real-time publisher
    for (unsigned i=0; i<forwardKinematicsComputer->getLinksNumber(); i++)
    {
      realtime_pub_->msg_.link_names.push_back("");
      realtime_pub_->msg_.link_poses.push_back(geometry_msgs::PoseStamped());
      realtime_pub_->msg_.link_twists.push_back(geometry_msgs::Twist());
    }

    ROS_INFO("Initializion of LinkStatePublisherController finished.");
    return true;
  }

  void LinkStatePublisherController::starting(const ros::Time& time)
  {
    // initialize time
    last_publish_time_ = time;
  }

  void LinkStatePublisherController::update(const ros::Time& time, const ros::Duration& /*period*/)
  {
    //ROS_INFO("LinkStatePublisherController update");
    // limit rate of publishing
    if (publish_rate_ > 0.0 && last_publish_time_ + ros::Duration(1.0/publish_rate_) < time){

      //ROS_INFO("LinkStatePublisherController update inner");
      // try to publish
      if (realtime_pub_->trylock()){
        // we're actually publishing, so increment time
        last_publish_time_ = last_publish_time_ + ros::Duration(1.0/publish_rate_);

        // populate joint state message:
        // - fill only joints that are present in the JointStateInterface, i.e. indices [0, num_hw_joints_)
        // - leave unchanged extra joints, which have static values, i.e. indices from num_hw_joints_ onwards
        joint_state_msg.header.stamp = time;
        for (unsigned i=0; i<num_hw_joints_; i++){
          joint_state_msg.position[i] = joint_state_[i].getPosition();
          joint_state_msg.velocity[i] = joint_state_[i].getVelocity();
          joint_state_msg.effort[i] = joint_state_[i].getEffort();
        }

        realtime_pub_->msg_ = forwardKinematicsComputer->computeLinkStates(joint_state_msg);
        realtime_pub_->unlockAndPublish();
        //ROS_INFO("Published link state");
      }
    }
  }

  void LinkStatePublisherController::stopping(const ros::Time& /*time*/)
  {}

  void LinkStatePublisherController::addExtraJoints(const ros::NodeHandle& nh, sensor_msgs::JointState& msg)
  {

    // Preconditions
    XmlRpc::XmlRpcValue list;
    if (!nh.getParam("extra_joints", list))
    {
      ROS_DEBUG("No extra joints specification found.");
      return;
    }

    if (list.getType() != XmlRpc::XmlRpcValue::TypeArray)
    {
      ROS_ERROR("Extra joints specification is not an array. Ignoring.");
      return;
    }

    for(int i = 0; i < list.size(); ++i)
    {
      XmlRpc::XmlRpcValue& elem = list[i];

      if (elem.getType() != XmlRpc::XmlRpcValue::TypeStruct)
      {
        ROS_ERROR_STREAM("Extra joint specification is not a struct, but rather '" << elem.getType() <<
                         "'. Ignoring.");
        continue;
      }

      if (!elem.hasMember("name"))
      {
        ROS_ERROR_STREAM("Extra joint does not specify name. Ignoring.");
        continue;
      }

      const std::string name = elem["name"];
      if (std::find(msg.name.begin(), msg.name.end(), name) != msg.name.end())
      {
        ROS_WARN_STREAM("Joint state interface already contains specified extra joint '" << name << "'.");
        continue;
      }

      const bool has_pos = elem.hasMember("position");
      const bool has_vel = elem.hasMember("velocity");
      const bool has_eff = elem.hasMember("effort");

      const XmlRpc::XmlRpcValue::Type typeDouble = XmlRpc::XmlRpcValue::TypeDouble;
      if (has_pos && elem["position"].getType() != typeDouble)
      {
        ROS_ERROR_STREAM("Extra joint '" << name << "' does not specify a valid default position. Ignoring.");
        continue;
      }
      if (has_vel && elem["velocity"].getType() != typeDouble)
      {
        ROS_ERROR_STREAM("Extra joint '" << name << "' does not specify a valid default velocity. Ignoring.");
        continue;
      }
      if (has_eff && elem["effort"].getType() != typeDouble)
      {
        ROS_ERROR_STREAM("Extra joint '" << name << "' does not specify a valid default effort. Ignoring.");
        continue;
      }

      // State of extra joint
      const double pos = has_pos ? static_cast<double>(elem["position"]) : 0.0;
      const double vel = has_vel ? static_cast<double>(elem["velocity"]) : 0.0;
      const double eff = has_eff ? static_cast<double>(elem["effort"])   : 0.0;

      // Add extra joints to message
      msg.name.push_back(name);
      msg.position.push_back(pos);
      msg.velocity.push_back(vel);
      msg.effort.push_back(eff);
    }
  }

}

PLUGINLIB_EXPORT_CLASS( lr_gym_utils::LinkStatePublisherController, controller_interface::ControllerBase)
