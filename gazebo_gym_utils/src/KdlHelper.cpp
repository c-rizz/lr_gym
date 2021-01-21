#include "../include/KdlHelper.hpp"
#include <ros/ros.h>
#include <urdf/model.h>
#include <utility>

namespace gazebo_gym_utils
{

bool KdlHelper::getSegmentForJoint(std::string jointName, KDL::Tree tree, KDL::Segment& segment)
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


bool KdlHelper::getSegmentParent(KDL::Segment& segment, KDL::Tree tree, KDL::Segment& parent)
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

KDL::Chain KdlHelper::getChainFromJoints(const KDL::Tree& tree, std::vector<std::string> jointNames)
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


KDL::Chain KdlHelper::getChainFromJoints(std::vector<std::string> jointNames, std::string robot_description_param_name)
{
  // gets the location of the robot description on the parameter server
  urdf::Model model;
  if (!model.initParam(robot_description_param_name))
    throw std::runtime_error("Failed to get robot description from parameter "+robot_description_param_name);

  KDL::Tree tree;
  if (!kdl_parser::treeFromUrdfModel(model, tree))
    throw std::runtime_error("Failed to extract kdl tree from xml robot description at param "+robot_description_param_name);

  try
  {
    KDL::Chain robotChain = getChainFromJoints(tree, jointNames);
    return robotChain;
  }
  catch(const std::runtime_error& e)
  {
    std::string err = std::string("Failed to get chain for the specified joints: ")+e.what();
    ROS_ERROR_STREAM(err);
    throw std::runtime_error(err);
  }

}

std::vector<std::pair<double, double>> getJointLimits(const urdf::Model& model, const std::vector<std::string>& jointNames)
{
  //See code at:
  //  https://github.com/ros/urdfdom_headers/blob/master/urdf_model/include/urdf_model/model.h
  //  https://github.com/ros/urdfdom_headers/blob/master/urdf_model/include/urdf_model/joint.h

  auto ret = std::vector<std::pair<double, double>>(jointNames.size());
  int i = 0;
  for(std::string jn : jointNames)
  {
    auto minMaxLim = std::pair<double,double>(model.joints_.at(jn)->limits->lower,
                                              model.joints_.at(jn)->limits->upper);
    ret[i++] = minMaxLim;
  }
  return ret;
}

std::vector<std::pair<double, double>> getJointLimits(std::string robot_description_param_name, const std::vector<std::string>& jointNames)
{
  urdf::Model model;
  if (!model.initParam(robot_description_param_name))
    throw std::runtime_error("Failed to get robot description from parameter "+robot_description_param_name);
  
  return getJointLimits(model, jointNames);
}

}
