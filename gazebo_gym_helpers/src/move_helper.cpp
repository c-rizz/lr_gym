#include <stdexcept>
#include <chrono>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>

#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>

#include <actionlib/server/simple_action_server.h>
#include <tf/tf.h>
#include <rviz_visual_tools/rviz_visual_tools.h>
#include <tf2_ros/transform_listener.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

#include <gazebo_gym_helpers/MoveToEePoseAction.h>
#include <gazebo_gym_helpers/MoveToJointPoseAction.h>
#include <gazebo_gym_helpers/GetJointState.h>
#include <gazebo_gym_helpers/GetEePose.h>


std::shared_ptr<actionlib::SimpleActionServer<gazebo_gym_helpers::MoveToEePoseAction>> moveToEePoseActionServer;
std::shared_ptr<actionlib::SimpleActionServer<gazebo_gym_helpers::MoveToJointPoseAction>> moveToJointPoseActionServer;
std::shared_ptr<moveit::planning_interface::MoveGroupInterface> moveGroupInt;
std::shared_ptr<tf2_ros::Buffer> tfBuffer;
std::string defaultEeLink = "";
const robot_state::JointModelGroup* joint_model_group;
std::string planning_group_name;


int waitActionCompletion(moveit::planning_interface::MoveGroupInterface& move_group)
{
  move_group.getMoveGroupClient().waitForResult(ros::Duration(0, 0));//TODO: set a sensible timeout
  if (move_group.getMoveGroupClient().getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
  {
    throw std::runtime_error(move_group.getMoveGroupClient().getState().toString() + ": " + move_group.getMoveGroupClient().getState().getText()+"\n"+
                            "Action execution failed with MoveItErrorCode "+std::to_string(move_group.getMoveGroupClient().getResult()->error_code.val));
  }

  ROS_INFO_STREAM("Moved.");
  return 0;
}


int submitMoveToJointPose(moveit::planning_interface::MoveGroupInterface& move_group, std::vector<double> jointPose)
{
  if(move_group.getVariableCount()!=jointPose.size())
    throw std::runtime_error("Provided joint pose has wrong size. Should be "+std::to_string(move_group.getVariableCount())+" it's "+std::to_string(jointPose.size()));
  move_group.setJointValueTarget(jointPose);

  ROS_INFO_STREAM("Joint pose target set.");
  auto r = move_group.asyncMove();
  if(r != moveit::planning_interface::MoveItErrorCode::SUCCESS)
    throw std::runtime_error("Pose-based asyncMove submission failed with MoveItErrorCode "+std::to_string(r.val));
  ROS_INFO_STREAM("Moving...");
  return 0;
}


int submitMoveToEePose(moveit::planning_interface::MoveGroupInterface& move_group, const geometry_msgs::PoseStamped& targetPose, std::string endEffectorLink)
{

  geometry_msgs::PoseStamped targetPoseBaseLink;
  try{
    targetPoseBaseLink = tfBuffer->transform(targetPose, "world", ros::Duration(1));
  }
  catch (tf2::TransformException &ex) {
    throw std::runtime_error("Failed to transform target ee pose to world: "+std::string(ex.what()));
  }

  move_group.clearPoseTargets();
  std::string eeLink = endEffectorLink;
  if(eeLink=="")
    eeLink = defaultEeLink;

  move_group.setPoseTarget(targetPoseBaseLink,endEffectorLink);

  ROS_INFO_STREAM("Pose target set.");
  auto r = move_group.asyncMove();
  if(r != moveit::planning_interface::MoveItErrorCode::SUCCESS)
    throw std::runtime_error("Pose-based asyncMove submission failed with MoveItErrorCode "+std::to_string(r.val));
  ROS_INFO_STREAM("Moving...");
  return 0;
}








int moveToJointPose(moveit::planning_interface::MoveGroupInterface& move_group, std::vector<double> jointPose)
{
  int r = submitMoveToJointPose(move_group,jointPose);
  if(r<0)
    throw std::runtime_error("Failed to submit move to joint pose, error "+r);

  r = waitActionCompletion(move_group);
  if(r<0)
    throw std::runtime_error("Failed to execute move to joint pose, error "+r);
  return 0;
}


int moveToEePose(moveit::planning_interface::MoveGroupInterface& move_group, const geometry_msgs::PoseStamped& targetPose, std::string endEffectorLink)
{
  int r = submitMoveToEePose(move_group, targetPose, endEffectorLink);
  if(r<0)
    throw std::runtime_error("Failed to submit move to end effectore pose, error "+r);

  r = waitActionCompletion(move_group);
  if(r<0)
    throw std::runtime_error("Failed to execute move to end effector pose, error "+r);
  return 0;
}

void moveToEePoseActionCallback(const gazebo_gym_helpers::MoveToEePoseGoalConstPtr &goal)
{
  try
  {
    moveToEePose(*moveGroupInt,goal->pose,goal->end_effector_link);
  }
  catch(std::runtime_error& e)
  {
    std::string errorMsg = "moveToEePose failed with error "+std::string(e.what());
    ROS_ERROR_STREAM(errorMsg);
    gazebo_gym_helpers::MoveToEePoseResult result;
    result.succeded = false;
    result.error_message = errorMsg;
    moveToEePoseActionServer->setAborted(result);
    return;
  }

  gazebo_gym_helpers::MoveToEePoseResult result;
  result.succeded = true;
  result.error_message = "No error";
  moveToEePoseActionServer->setSucceeded(result);

}


void moveToJointPoseActionCallback(const gazebo_gym_helpers::MoveToJointPoseGoalConstPtr &goal)
{
  try
  {
    moveToJointPose(*moveGroupInt,goal->pose);
  }
  catch(std::runtime_error& e)
  {
    std::string errorMsg = "moveToJointPose failed with error "+std::string(e.what());
    ROS_ERROR_STREAM(errorMsg);
    gazebo_gym_helpers::MoveToJointPoseResult result;
    result.succeded = false;
    result.error_message = errorMsg;
    moveToJointPoseActionServer->setAborted(result);
    return;
  }

  gazebo_gym_helpers::MoveToJointPoseResult result;
  result.succeded = true;
  result.error_message = "No error";
  moveToJointPoseActionServer->setSucceeded(result);

}


bool getJointStateServiceCallback(gazebo_gym_helpers::GetJointState::Request& req, gazebo_gym_helpers::GetJointState::Response& res)
{
  ROS_INFO("Getting joint state...");
  moveGroupInt->getCurrentState(10)->copyJointGroupPositions(joint_model_group, res.joint_poses);
  ROS_INFO("Got joint state.");
  return true;
}

bool getEePoseServiceCallback(gazebo_gym_helpers::GetEePose::Request& req, gazebo_gym_helpers::GetEePose::Response& res)
{
  ROS_INFO("Getting end effector pose...");
  res.pose = moveGroupInt->getCurrentPose(req.end_effector_link_name);
  ROS_INFO("Got end effector pose...");
  return true;
}



int main(int argc, char** argv)
{
  ros::init(argc, argv, "move_helper");
  ros::NodeHandle node_handle("~");

  planning_group_name = "panda_arm";
  defaultEeLink = "panda_link8";
  ros::AsyncSpinner spinner(2);
  spinner.start();

  ROS_INFO("Creating MoveGroupInterface...");
  moveGroupInt = std::make_shared<moveit::planning_interface::MoveGroupInterface>(planning_group_name,std::shared_ptr<tf2_ros::Buffer>(),ros::WallDuration(30));
  ROS_INFO("MoveGroupInterface created.");
  joint_model_group = moveGroupInt->getCurrentState()->getJointModelGroup(planning_group_name);

  moveGroupInt->setPlanningTime(0.15);

  tfBuffer = std::make_shared<tf2_ros::Buffer>();
  tf2_ros::TransformListener tfListener(*tfBuffer);


  moveToEePoseActionServer = std::make_shared<actionlib::SimpleActionServer<gazebo_gym_helpers::MoveToEePoseAction>>(node_handle,
                                                                                                                  "move_to_ee_pose",
                                                                                                                  moveToEePoseActionCallback,
                                                                                                                  false);
  moveToEePoseActionServer->start();

  moveToJointPoseActionServer = std::make_shared<actionlib::SimpleActionServer<gazebo_gym_helpers::MoveToJointPoseAction>>(node_handle,
                                                                                                                        "move_to_joint_pose",
                                                                                                                        moveToJointPoseActionCallback,
                                                                                                                        false);
  moveToJointPoseActionServer->start();

  ros::ServiceServer service = node_handle.advertiseService("get_joint_state", getJointStateServiceCallback);
  ros::ServiceServer getEePoseService = node_handle.advertiseService("get_ee_pose", getEePoseServiceCallback);

  ROS_INFO("Action and service servers started");
  ros::waitForShutdown();
  return 0;
}
