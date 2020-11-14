#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <urdf/model.h>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainfksolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainfksolvervel_recursive.hpp>
#include <kdl_conversions/kdl_msg.h>
#include <gazebo_gym_utils/LinkStates.h>
#include <geometry_msgs/PoseArray.h>
#include "../include/ForwardKinematicsComputer.hpp"

ros::Publisher linkStatesDbgPublisher;
std::shared_ptr<gazebo_gym_utils::ForwardKinematicsComputer> forwardKinematicsComputer;


void jointStatesCallback(const gazebo_gym_utils::LinkStatesConstPtr& msg)
{

  geometry_msgs::PoseArray poseArrayDbg = forwardKinematicsComputer->getLinkPoses(*msg);

  linkStatesDbgPublisher.publish(poseArrayDbg);
}



int main(int argc, char** argv)
{
  ros::init(argc, argv, "link_pose_publisher");
  ros::NodeHandle node_handle;

  forwardKinematicsComputer = std::make_shared<gazebo_gym_utils::ForwardKinematicsComputer>();

  ros::Subscriber jointStatesSub = node_handle.subscribe("link_states", 1, jointStatesCallback);

  linkStatesDbgPublisher = node_handle.advertise<geometry_msgs::PoseArray>("link_poses_dbg", 1);

  ROS_INFO("Link pose publisher started");
  ros::spin();
  return 0;
}
