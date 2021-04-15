#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <urdf/model.h>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainfksolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainfksolvervel_recursive.hpp>
#include <kdl_conversions/kdl_msg.h>
#include <lr_gym_utils/LinkStates.h>
#include <geometry_msgs/PoseArray.h>
#include "../include/ForwardKinematicsComputer.hpp"

ros::Publisher linkStatesPublisher;
ros::Publisher linkStatesDbgPublisher;
std::shared_ptr<lr_gym_utils::ForwardKinematicsComputer> forwardKinematicsComputer;


void jointStatesCallback(const sensor_msgs::JointStateConstPtr& msg)
{

  lr_gym_utils::LinkStates linkStates = forwardKinematicsComputer->computeLinkStates(*msg);
  geometry_msgs::PoseArray poseArrayDbg = forwardKinematicsComputer->getLinkPoses(linkStates);

  linkStatesPublisher.publish(linkStates);
  linkStatesDbgPublisher.publish(poseArrayDbg);
}



int main(int argc, char** argv)
{
  ros::init(argc, argv, "link_states_publisher");
  ros::NodeHandle node_handle;

  forwardKinematicsComputer = std::make_shared<lr_gym_utils::ForwardKinematicsComputer>();

  ros::Subscriber jointStatesSub = node_handle.subscribe("joint_states", 1, jointStatesCallback);

  linkStatesPublisher = node_handle.advertise<lr_gym_utils::LinkStates>("link_states", 1);
  linkStatesDbgPublisher = node_handle.advertise<geometry_msgs::PoseArray>("link_poses_dbg", 1);

  ROS_INFO("Link state publisher started");
  ros::spin();
  return 0;
}
