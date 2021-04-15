#include "../include/ForwardKinematicsComputer.hpp"
#include <stdexcept>

namespace gazebo_gym_utils
{

  gazebo_gym_utils::LinkStates ForwardKinematicsComputer::computeLinkStates(const sensor_msgs::JointState& msg)
  {
    //ROS_INFO("got joints");
    std::map<std::string,FkResult> fkResults;

    // See: https://www.orocos.org/kdl/examples#comment-964
    for(KDL::Chain chain : chains)
    {
      //ROS_DEBUG_STREAM("Computing FK");
      std::string tipName = chain.segments.back().getName();
      //ROS_DEBUG_STREAM("Computing pose for link "<<tipName);

      unsigned int nrOfJoints = chain.getNrOfJoints();
      KDL::JntArray jointPositions = KDL::JntArray(nrOfJoints);
      KDL::JntArray jointVelocities = KDL::JntArray(nrOfJoints);

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
          auto jointIt = std::find(msg.name.begin(),msg.name.end(),jointName);
          if(jointIt != msg.name.end())
          {
            double jointPosition = msg.position.at(jointIt - msg.name.begin());
            double jointVelocity = msg.velocity.at(jointIt - msg.name.begin());
            //ROS_DEBUG_STREAM("Got joint position ("<<foundJoints<<" of "<<nrOfJoints<<")");
            jointPositions(foundJoints) = jointPosition;
            jointVelocities(foundJoints) = jointVelocity;
            foundJoints++;
          }
          else
          {
            ROS_DEBUG_STREAM("Couldn't find position of joint "<<jointName<<" in joint_state message. Will skip forward kinematics for its chain(s).");
            break;
          }
        }
      }
      if(foundJoints!=nrOfJoints)
      {
        ROS_DEBUG_STREAM("Couldn't find all joint positions, skipping chain for "<<tipName);
        continue;
      }
      KDL::JntArrayVel jointPosAndVel(jointPositions,jointVelocities);



      //ROS_DEBUG_STREAM("Computing forward kinematics");
      KDL::Frame cartesianPosition;
      KDL::ChainFkSolverPos_recursive fksolver = KDL::ChainFkSolverPos_recursive(chain);
      int ret = fksolver.JntToCart(jointPositions,cartesianPosition);
      if(ret<0)
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
          jointPositionsStr += std::to_string(jointPositions(i)) + ", ";
        ROS_WARN_STREAM("Forward kinematics for chain failed. Will skip chain for link "<<tipName<<". \n Links: ("+linkNamesStr+")\n Joints: ("+jointNamesStr+")\n Positions: ("+jointPositionsStr+")");
        continue;
      }

      KDL::ChainFkSolverVel_recursive fkVelSolver = KDL::ChainFkSolverVel_recursive(chain);
      KDL::FrameVel cartesianVelocity;
      ret = fkVelSolver.JntToCart(jointPosAndVel, cartesianVelocity);
      if(ret<0)
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
          jointPositionsStr += std::to_string(jointPositions(i)) + ", ";
        ROS_WARN_STREAM("Forward kinematics velocity computation for chain failed. Will skip chain for link "<<tipName<<". \n Links: ("+linkNamesStr+")\n Joints: ("+jointNamesStr+")\n Positions: ("+jointPositionsStr+")");
        continue;
      }
      //ROS_DEBUG_STREAM("Computed forward kinematics");

      FkResult res;
      res.pose = cartesianPosition;
      res.twist = cartesianVelocity.GetTwist();
      fkResults.insert(std::pair<std::string,FkResult>(tipName, res));
      //ROS_DEBUG_STREAM("Saved result");
    }

    //ROS_DEBUG_STREAM("Computed forward kinematics for "<<linkCartesianPoses.size()<<" links");
    gazebo_gym_utils::LinkStates linkStates;

    for(std::pair<std::string,FkResult> fkResult : fkResults)
    {
      geometry_msgs::PoseStamped linkPose;
      tf::poseKDLToMsg(fkResult.second.pose,linkPose.pose);
      linkPose.header.frame_id = rootLinkName;
      linkPose.header.stamp = msg.header.stamp;
      geometry_msgs::Twist linkTwist;
      tf::twistKDLToMsg(fkResult.second.twist,linkTwist);

      linkStates.link_names.push_back(fkResult.first);
      linkStates.link_poses.push_back(linkPose);
      linkStates.link_twists.push_back(linkTwist);
    }
    linkStates.header.stamp = msg.header.stamp;

    return linkStates;
  }

  geometry_msgs::PoseArray ForwardKinematicsComputer::getLinkPoses(const gazebo_gym_utils::LinkStates& linkStates)
  {
    geometry_msgs::PoseArray poseArray;
    poseArray.header.frame_id = rootLinkName;
    poseArray.header.stamp = linkStates.header.stamp;
    for(geometry_msgs::PoseStamped linkPose : linkStates.link_poses)
      poseArray.poses.push_back(linkPose.pose);
    return poseArray;
  }

  std::vector<KDL::Chain> ForwardKinematicsComputer::treeToChains(const KDL::Tree& tree)
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

  unsigned int ForwardKinematicsComputer::getLinksNumber()
  {
    return chains.size();
  }


  ForwardKinematicsComputer::ForwardKinematicsComputer()
  {

    // gets the location of the robot description on the parameter server
    urdf::Model model;
    if (!model.initParam("robot_description"))
      throw std::runtime_error("Failed to get robot_description");

    KDL::Tree tree;
    if (!kdl_parser::treeFromUrdfModel(model, tree))
    {
      throw std::runtime_error("Failed to extract kdl tree from xml robot description");
    }

    rootLinkName = tree.getRootSegment()->second.segment.getName();
    chains = treeToChains(tree);

  }

}
