#pragma once

#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl_conversions/kdl_msg.h>
#include <kdl/chaindynparam.hpp>
#include <urdf/model.h>

namespace gazebo_gym_utils
{

class KdlHelper
{
public:
    static bool getSegmentForJoint(std::string jointName, KDL::Tree tree, KDL::Segment& segment);
    static bool getSegmentParent(KDL::Segment& segment, KDL::Tree tree, KDL::Segment& parent);
    static KDL::Chain getChainFromJoints(const KDL::Tree& tree, std::vector<std::string> jointNames);
    static KDL::Chain getChainFromJoints(std::vector<std::string> jointNames, std::string robot_description_param_name = "robot_description");
    static std::vector<std::pair<double, double>> getJointLimits(std::string robot_description_param_name, const std::vector<std::string>& jointNames);
    static std::vector<std::pair<double, double>> getJointLimits(const urdf::Model& model, const std::vector<std::string>& jointNames);
};

}
