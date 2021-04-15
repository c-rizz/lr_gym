
#include "../include/CartesianPositionController.hpp"
#include "../include/KdlHelper.hpp"
#include <pluginlib/class_list_macros.h>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_pinv_givens.hpp>
#include <random>

namespace lr_gym_utils
{

  CartesianPositionController::CartesianPositionController()
  {

  }





  CartesianPositionController::~CartesianPositionController()
  {
    commandSubscriber.shutdown();
  }








  bool CartesianPositionController::init(hardware_interface::PositionJointInterface* hw, ros::NodeHandle &n)
  {
    ROS_INFO("Initializing CartesianPositionController");
    // List of controlled joints
    std::string joint_chain_param_name = "joint_chain";
    std::string robot_description_param_name = "robot_description";
    std::string attempts_param_name = "attempts";
    std::string iterations_param_name = "iterations";
    std::string precision_param_name = "precision";
    std::vector<std::string> joint_names;
    if(!n.getParam(joint_chain_param_name, joint_names))
    {
      ROS_ERROR_STREAM("GravityCompensatedEffortController: Failed to getParam '" << joint_chain_param_name << "' (namespace: " << n.getNamespace() << ").");
      return false;
    }

    if(joint_names.size() == 0)
    {
      ROS_ERROR_STREAM("GravityCompensatedEffortController: List of joint names is empty.");
      return false;
    }

    if(!n.getParam(attempts_param_name, attempts))
    {
      ROS_ERROR_STREAM("GravityCompensatedEffortController: Failed to getParam '" << attempts_param_name << "' (namespace: " << n.getNamespace() << ").");
      return false;
    }

    if(!n.getParam(iterations_param_name, iterations))
    {
      ROS_ERROR_STREAM("GravityCompensatedEffortController: Failed to getParam '" << iterations_param_name << "' (namespace: " << n.getNamespace() << ").");
      return false;
    }

    if(!n.getParam(precision_param_name, precision))
    {
      ROS_ERROR_STREAM("GravityCompensatedEffortController: Failed to getParam '" << precision_param_name << "' (namespace: " << n.getNamespace() << ").");
      return false;
    }
    


    std::string joint_names_str = "";
    for(std::string jn : joint_names)
      joint_names_str+=jn +", ";
    ROS_INFO_STREAM("GravityCompensatedEffortController: Got joint_names = ["<<joint_names_str<<"]");

    try
    {
      robotChain = KdlHelper::getChainFromJoints(joint_names, robot_description_param_name);
    }
    catch(const std::runtime_error& e)
    {
      ROS_ERROR_STREAM("GravityCompensatedEffortController: Failed to get chain for the specified joints: "<<e.what());
      return false;
    }


    ROS_INFO_STREAM("GravityCompensatedEffortController: Built chain with segments:");
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
        joint_handles.push_back(hw->getHandle(jn));
      }
      catch (const hardware_interface::HardwareInterfaceException& e)
      {
        ROS_ERROR_STREAM("GravityCompensatedEffortController: Exception thrown: " << e.what());
        return false;
      }
    }


    std::vector<std::pair<double, double>> jointLimits = KdlHelper::getJointLimits(robot_description_param_name, notFixedJointsNames);

    joint_limits_low = KDL::JntArray(notFixedJointsNames.size());
    joint_limits_high = KDL::JntArray(notFixedJointsNames.size());
    for(unsigned int i = 0 ; i<jointLimits.size(); i++)
    {
      joint_limits_low(i) = jointLimits[i].first;
      joint_limits_high(i) = jointLimits[i].second;
    }

    std::random_device rd;
    randomEngine = std::mt19937(rd());

    lastSuccessfulCommand = std::vector<double>(7);
    command_buffer.writeFromNonRT(computeFk(getCurrentJointPose()));
    commandSubscriber = n.subscribe<std_msgs::Float64MultiArray>("command", 1, &CartesianPositionController::commandCB, this);
    ROS_INFO("Initialization of CartesianPositionController finished.");
    return true;
  }




  std::vector<double> CartesianPositionController::getCurrentJointPose()
  {
    std::vector<double> joint_position = std::vector<double>(notFixedJointsNames.size());
    int i=0;
    for(hardware_interface::JointHandle jh : joint_handles)
      joint_position[i++] = jh.getPosition();
    return joint_position;
  }






  void CartesianPositionController::starting(const ros::Time& time)
  {
    // Start controller with curent pose
    *(command_buffer.readFromRT()) = computeFk(getCurrentJointPose());
  }










  void CartesianPositionController::commandCB(const std_msgs::Float64MultiArrayConstPtr& msg)
  {
    if(msg->data.size()!=7)
    {
      ROS_ERROR_STREAM("Dimension of command (" << msg->data.size() << ") does not match size. Should be (pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w). Ignoring command.");
      return;
    }
    command_buffer.writeFromNonRT(msg->data);
  }



  std::vector<double> CartesianPositionController::computeFk(std::vector<double> jointPose)
  {
    if(jointPose.size()!=robotChain.getNrOfJoints())
      throw std::runtime_error("CartesianPositionController::computeFk: invalid number of joints provided, got "+std::to_string(jointPose.size())+", should be "+std::to_string(robotChain.getNrOfJoints()));

    ROS_INFO_STREAM("Computing fk for "<<poseToStr(jointPose));
    KDL::JntArray jointPositions = KDL::JntArray(robotChain.getNrOfJoints());
    for(unsigned int i=0; i<jointPose.size(); i++)
      jointPositions(i) = jointPose[i];
    KDL::Frame cartesianPose;
    KDL::ChainFkSolverPos_recursive fkpossolver(robotChain);
    int status = fkpossolver.JntToCart(jointPositions,cartesianPose);
    if(status<0)
      throw std::runtime_error("CartesianPositionController::computeFk: Failed to compute forward kinematics for joint pose "+poseToStr(jointPose));

    std::vector<double> ret(7);
    ret[0] = cartesianPose.p.x();
    ret[1] = cartesianPose.p.y();
    ret[2] = cartesianPose.p.z();
    cartesianPose.M.GetQuaternion(ret[3],ret[4],ret[5],ret[6]);
    ROS_INFO_STREAM("Fk computed "<<poseToStr(ret));
    return ret;
  }


  std::vector<double> CartesianPositionController::computeIk(std::vector<double> requestedCartesianPose)
  {
    ROS_INFO_STREAM("Computing ik for "<<poseToStr(requestedCartesianPose));
    KDL::ChainFkSolverPos_recursive fkpossolver(robotChain);
    KDL::ChainIkSolverVel_pinv_givens ikvelsolver(robotChain);
    KDL::ChainIkSolverPos_NR_JL inv_pos_solver(robotChain,
                                               joint_limits_low,joint_limits_high,
                                               fkpossolver,ikvelsolver,
                                               iterations,precision);

    KDL::Frame requestedCartesianPose_frame = KDL::Frame(KDL::Rotation::Quaternion( requestedCartesianPose[3],
                                                                                    requestedCartesianPose[4],
                                                                                    requestedCartesianPose[5],
                                                                                    requestedCartesianPose[6]),
                                                         KDL::Vector(requestedCartesianPose[0],
                                                                     requestedCartesianPose[1],
                                                                     requestedCartesianPose[2]));

    KDL::JntArray jointPose(robotChain.getNrOfJoints());

    std::vector<double> currentJointPose_vec = getCurrentJointPose();
    KDL::JntArray currentJointPose(robotChain.getNrOfJoints());
    for(unsigned int i=0; i < robotChain.getNrOfJoints(); i++ )
        currentJointPose(i) = currentJointPose_vec.at(i);

    int attempt=0;
    for(; attempt<attempts; attempt++)
    {

      KDL::JntArray q_init(robotChain.getNrOfJoints());
      if(attempt == 0)
        q_init = currentJointPose;
      else
        q_init = randomJointPose();
      int status = inv_pos_solver.CartToJnt(q_init,
                                            requestedCartesianPose_frame,
                                            jointPose);
      if(status >= 0)
        break;
      if(status!=-5)
        throw std::runtime_error("Failed to compute IK for cartesian pose "+poseToStr(requestedCartesianPose)+". status = "+std::to_string(status));
    }
    if(attempt>=attempts)
      throw std::runtime_error("Failed to compute IK for cartesian pose "+poseToStr(requestedCartesianPose)+". Too many attempts.");
    
    std::vector<double> jointPoseVec(robotChain.getNrOfJoints());
    for(unsigned int i=0;i<jointPoseVec.size();i++)
      jointPoseVec[i] = jointPose(i);
    ROS_INFO_STREAM("ik result is "<<poseToStr(jointPoseVec));
    return jointPoseVec;
  }

  KDL::JntArray CartesianPositionController::randomJointPose()
  {
    KDL::JntArray p(robotChain.getNrOfJoints());
    for(unsigned int i=0; i<robotChain.getNrOfJoints(); i++)
    {
      std::uniform_real_distribution<> dist(joint_limits_low(i)+0.00001, joint_limits_high(i));
      p(i) = dist(randomEngine);
    }
    return p;
  }

  std::string CartesianPositionController::poseToStr(std::vector<double> p)
  {
    std::string s = "[";
    for(double v : p)
      s += std::to_string(v)+" ";
    s += "]";
    return s;
  }



  void CartesianPositionController::update(const ros::Time& /*time*/, const ros::Duration& /*period*/)
  {
    std::vector<double> requestedCartesianPose = *command_buffer.readFromRT();
    bool isNew = false;
    for(unsigned int i=0; i<requestedCartesianPose.size(); i++)
    {
      if(requestedCartesianPose.at(i)!=lastSuccessfulCommand.at(i))
        isNew=true;
    }
    if(isNew)
    {
      std::vector<double> jointPose;
      try
      {
        jointPose = computeIk(requestedCartesianPose);
      }
      catch(const std::runtime_error& e)
      {
        std::string err = std::string("Failed to compute inverse kinematics: ")+e.what();
        ROS_ERROR_STREAM(err);
        return;
      }

      lastSuccessfulCommand = requestedCartesianPose;
      lastJointPose = jointPose;
    }
    setJointPose(lastJointPose);
  }

  void CartesianPositionController::setJointPose(std::vector<double> jointPose)
  {
    for(unsigned int i=0; i<notFixedJointsNames.size(); i++)
      joint_handles[i].setCommand(jointPose[i]);
  }
}


PLUGINLIB_EXPORT_CLASS(lr_gym_utils::CartesianPositionController, controller_interface::ControllerBase)
