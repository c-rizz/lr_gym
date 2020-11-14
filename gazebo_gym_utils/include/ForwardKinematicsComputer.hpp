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

namespace gazebo_gym_utils
{
  class ForwardKinematicsComputer
  {
  private:
    std::vector<KDL::Chain> chains;
    std::string rootLinkName;

    struct FkResult
    {
      KDL::Frame pose;
      KDL::Twist twist;
    };



    std::vector<KDL::Chain> treeToChains(const KDL::Tree& tree);

  public:
    gazebo_gym_utils::LinkStates computeLinkStates(const sensor_msgs::JointState& msg);
    geometry_msgs::PoseArray getLinkPoses(const gazebo_gym_utils::LinkStates& linkStates);
    unsigned int getLinksNumber();
    ForwardKinematicsComputer();
  };
}
