#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <urdf/model.h>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainfksolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl_conversions/kdl_msg.h>
#include <gazebo_gym_helpers/LinkPoses.h>
#include <geometry_msgs/PoseArray.h>


std::vector<KDL::Chain> chains;
ros::Publisher linkPosesPublisher;
ros::Publisher linkPosesDbgPublisher;
std::string rootLinkName;

void jointStatesCallback(const sensor_msgs::JointStateConstPtr& msg)
{
  //ROS_INFO("got joints");
  std::map<std::string,KDL::Frame> linkCartesianPoses;

  // See: https://www.orocos.org/kdl/examples#comment-964
  for(KDL::Chain chain : chains)
  {
    //ROS_DEBUG_STREAM("Computing FK");
    std::string tipName = chain.segments.back().getName();
    //ROS_DEBUG_STREAM("Computing pose for link "<<tipName);
    KDL::ChainFkSolverPos_recursive fksolver = KDL::ChainFkSolverPos_recursive(chain);

    unsigned int nrOfJoints = chain.getNrOfJoints();
    KDL::JntArray jointPositions = KDL::JntArray(nrOfJoints);

    //ROS_DEBUG_STREAM("Getting joint positions");
    unsigned int foundJoints = 0;
    for(KDL::Segment& seg : chain.segments)
    {
      std::string jointName = seg.getJoint().getName();
      //ROS_DEBUG_STREAM("Getting position of joint "<<jointName);
      if(seg.getJoint().getType() != KDL::Joint::None)
      {
        // None means Rigid connection, and those are not counted as joints
        // https://orocos.org/wiki/main-page/kdl-wiki/user-manual/kinematic-trees/kinematic-trees-kdl-10x#toc45
        // http://docs.ros.org/jade/api/orocos_kdl/html/classKDL_1_1Chain.html#a99a689f48b21f015099baf65c6bf0c5b

        //ROS_DEBUG_STREAM("Not a fixed joint");
        auto jointIt = std::find(msg->name.begin(),msg->name.end(),jointName);
        if(jointIt != msg->name.end())
        {
          double jointPosition = msg->position.at(jointIt - msg->name.begin());
          //ROS_DEBUG_STREAM("Got joint position ("<<foundJoints<<" of "<<nrOfJoints<<")");
          jointPositions(foundJoints) = jointPosition;
          foundJoints++;
        }
        else
        {
          ROS_WARN_STREAM("Couldn't find position of joint "<<jointName<<" in joint_state message. Will skip forward kinematics for its chain(s).");
          break;
        }
      }
    }
    if(foundJoints!=nrOfJoints)
    {
      ROS_WARN_STREAM("Couldn't find all joint positions, skipping chain for "<<tipName);
      continue;
    }


    {
      std::string linkNamesStr = "";
      std::string jointNamesStr = "";
      for(KDL::Segment& seg : chain.segments)
      {
        linkNamesStr += seg.getName() + ", ";
        jointNamesStr += seg.getJoint().getName() + ", ";
      }
      std::string jointPositionsStr = "";
      for(unsigned int i = 0; i<nrOfJoints; i++)
      {
        jointPositionsStr += std::to_string(jointPositions(i)) + ", ";
      }
      //ROS_DEBUG_STREAM("computing forward kinematics. \n Links: ("+linkNamesStr+")\n Joints: ("+jointNamesStr+")\n Positions: ("+jointPositionsStr+")");
    }

    //ROS_DEBUG_STREAM("Computing forward kinematics");
    KDL::Frame cartesianPosition;
    int kinematics_status = fksolver.JntToCart(jointPositions,cartesianPosition);
    if(kinematics_status<0)
    {
      std::string linkNamesStr = "";
      std::string jointNamesStr = "";
      for(KDL::Segment& seg : chain.segments)
      {
        linkNamesStr += seg.getName() + ", ";
        jointNamesStr += seg.getJoint().getName() + ", ";
      }
      std::string jointPositionsStr = "";
      for(unsigned int i = 0; i<nrOfJoints; i++)
      {
        jointPositionsStr += std::to_string(jointPositions(i)) + ", ";
      }
      ROS_WARN_STREAM("Forward kinematics for chain failed. Will skip chain for link "<<tipName<<". \n Links: ("+linkNamesStr+")\n Joints: ("+jointNamesStr+")\n Positions: ("+jointPositionsStr+")");
      continue;
    }
    //ROS_DEBUG_STREAM("Computed forward kinematics");

    linkCartesianPoses.insert(std::pair<std::string,KDL::Frame>(tipName,cartesianPosition));

    //ROS_DEBUG_STREAM("Saved result");
  }

  //ROS_DEBUG_STREAM("Computed forward kinematics for "<<linkCartesianPoses.size()<<" links");
  gazebo_gym_helpers::LinkPoses linkPoses;
  geometry_msgs::PoseArray poseArrayDbg;

  for(std::pair<std::string,KDL::Frame> fkResult : linkCartesianPoses)
  {
    geometry_msgs::PoseStamped linkPose;
    tf::poseKDLToMsg(fkResult.second,linkPose.pose);
    linkPose.header.frame_id = rootLinkName;
    linkPose.header.stamp = msg->header.stamp;
    linkPoses.link_names.push_back(fkResult.first);
    linkPoses.link_poses.push_back(linkPose);
    poseArrayDbg.poses.push_back(linkPose.pose);
  }
  poseArrayDbg.header.frame_id = rootLinkName;
  poseArrayDbg.header.stamp = msg->header.stamp;

  linkPosesPublisher.publish(linkPoses);
  linkPosesDbgPublisher.publish(poseArrayDbg);
}

std::vector<KDL::Chain> treeToChains(const KDL::Tree& tree)
{
  //Get all the root segments (the chains have to start from the second segment as they all have a parent joint)
  std::vector<std::string> chainsRootSegmentNames;
  for(auto it : tree.getSegments().at(tree.getRootSegment()->second.segment.getName()).children)
    chainsRootSegmentNames.push_back(it->second.segment.getName());

  //Get all the chains that start from all the roots
  std::vector<KDL::Chain> ret;
  for(std::string chainsRootSegmentName : chainsRootSegmentNames)
  {
    KDL::SegmentMap segments = tree.getSegments();
    std::vector<std::string> segmentNames;
    for(auto it : segments)
    {
      //if(it.second.children.empty())
      //{
        segmentNames.push_back(it.second.segment.getName());
      //}
    }


    for(std::string leaveName : segmentNames)
    {
      const std::string tipName = leaveName;
      if(tipName==chainsRootSegmentName)
        continue;
      KDL::Chain chain;
      bool r = tree.getChain(chainsRootSegmentName,tipName,chain);
      if(!r)
      {
        std::string err = "KDL::Tree::getChain failed";
        ROS_ERROR_STREAM(err);
        throw std::runtime_error(err);
      }

      ROS_DEBUG_STREAM("Built chain with segments:");
      for(KDL::Segment& seg : chain.segments)
        ROS_DEBUG_STREAM(" - \""<<seg.getName()<<"\" (joint = \""<<seg.getJoint().getName()<<"\" of type "<<seg.getJoint().getTypeName()<<")");

      ret.push_back(chain);
    }
  }
  return ret;
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "link_states_publisher");
  ros::NodeHandle node_handle;

  // gets the location of the robot description on the parameter server
  urdf::Model model;
  if (!model.initParam("robot_description"))
    return 1;

  KDL::Tree tree;
  if (!kdl_parser::treeFromUrdfModel(model, tree))
  {
    ROS_ERROR("Failed to extract kdl tree from xml robot description");
    return 1;
  }

  rootLinkName = tree.getRootSegment()->second.segment.getName();
  chains = treeToChains(tree);

  ros::Subscriber jointStatesSub = node_handle.subscribe("joint_states", 1, jointStatesCallback);

  linkPosesPublisher = node_handle.advertise<gazebo_gym_helpers::LinkPoses>("link_states", 1);
  linkPosesDbgPublisher = node_handle.advertise<geometry_msgs::PoseArray>("link_poses_dbg", 1);

  ROS_INFO("Link state publisher started");
  ros::spin();
  return 0;
}
