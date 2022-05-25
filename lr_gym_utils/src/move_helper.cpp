#include <stdexcept>
#include <chrono>
#include <cmath>

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

#include <lr_gym_utils/MoveToEePoseAction.h>
#include <lr_gym_utils/MoveToJointPoseAction.h>
#include <lr_gym_utils/GetJointState.h>
#include <lr_gym_utils/GetEePose.h>
#include <lr_gym_utils/AddCollisionBox.h>
#include <lr_gym_utils/ClearCollisionObjects.h>


std::shared_ptr<actionlib::SimpleActionServer<lr_gym_utils::MoveToEePoseAction>> moveToEePoseActionServer;
std::shared_ptr<actionlib::SimpleActionServer<lr_gym_utils::MoveToJointPoseAction>> moveToJointPoseActionServer;
std::shared_ptr<moveit::planning_interface::MoveGroupInterface> moveGroupInt;
std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planningSceneInt;
std::vector<std::string> collision_objects_ids;
std::vector<std::string> attached_objects_ids;
std::shared_ptr<tf2_ros::Buffer> tfBuffer;
std::string defaultEeLink = "";
const robot_state::JointModelGroup* joint_model_group;
std::string planning_group_name;



int waitActionCompletion(moveit::planning_interface::MoveGroupInterface& move_group)
{
  move_group.getMoveGroupClient().waitForResult(ros::Duration(0, 0));//TODO: set a sensible timeout
  if (move_group.getMoveGroupClient().getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
  {
    throw std::runtime_error(move_group.getMoveGroupClient().getState().toString() + ": " + move_group.getMoveGroupClient().getState().getText()+". "+
                            "Action execution failed with MoveItErrorCode "+std::to_string(move_group.getMoveGroupClient().getResult()->error_code.val));
  }

  // ROS_INFO_STREAM("Moved.");
  return 0;
}


int submitMoveToJointPose(moveit::planning_interface::MoveGroupInterface& move_group, std::vector<double> jointPose)
{
  if(move_group.getVariableCount()!=jointPose.size())
    throw std::runtime_error("Provided joint pose has wrong size. Should be "+std::to_string(move_group.getVariableCount())+" it's "+std::to_string(jointPose.size()));
  move_group.setJointValueTarget(jointPose);

  // ROS_INFO_STREAM("Joint pose target set.");
  auto r = move_group.asyncMove();
  if(r != moveit::planning_interface::MoveItErrorCode::SUCCESS)
    throw std::runtime_error("submitMoveToJointPose(): asyncMove submission failed with MoveItErrorCode "+std::to_string(r.val));
  // ROS_INFO_STREAM("Moving...");
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

  // ROS_INFO_STREAM("Pose target set.");
  auto r = move_group.asyncMove();
  if(r != moveit::planning_interface::MoveItErrorCode::SUCCESS)
    throw std::runtime_error("submitMoveToEePose(): asyncMove submission failed with MoveItErrorCode "+std::to_string(r.val));
  // ROS_INFO_STREAM("Moving...");
  return 0;
}


int executeMoveToEePoseCartesian(moveit::planning_interface::MoveGroupInterface& move_group,
                                const geometry_msgs::PoseStamped& targetPose,
                                std::string endEffectorLink,
                                double velocity_scaling = 0.1)
{

  if(velocity_scaling<=0 || velocity_scaling>1)
    throw std::runtime_error("executeMoveToEePoseCartesian(): Invalid velocity scaling = "+std::to_string(velocity_scaling));
  // ROS_INFO_STREAM("Submitting Cartesian Move goal");
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


  move_group.setEndEffectorLink(eeLink);
  move_group.setPoseReferenceFrame("world");

  geometry_msgs::PoseStamped startpose = moveGroupInt->getCurrentPose(eeLink);
  std::vector<double> diff = {startpose.pose.position.x - targetPose.pose.position.x,
                              startpose.pose.position.y - targetPose.pose.position.y,
                              startpose.pose.position.z - targetPose.pose.position.z};
  double distance = sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);


  const double jump_threshold = 1.5; // No joint-space jump contraint (see moveit_msgs/GetCartesianPath)
  double eef_step = 0.001;
  if (distance/eef_step < 15)
    eef_step = distance/15;

  moveit_msgs::RobotTrajectory robotTraj;
  std::vector<geometry_msgs::Pose> waypoints;
  waypoints.push_back(targetPoseBaseLink.pose);
  double fraction = moveGroupInt->computeCartesianPath(waypoints, eef_step, jump_threshold, robotTraj);
  if(fraction < 0.99)
    throw std::runtime_error("planning on group "+moveGroupInt->getName()+" for end effector "+eeLink+" failed with fraction "+std::to_string(fraction));
  
  double scaling_factor = velocity_scaling;
  // ROS_INFO_STREAM("Scaling cartesian path by "<<scaling_factor);
  for(unsigned int i=0;i<robotTraj.joint_trajectory.points.size();i++)
  {
    trajectory_msgs::JointTrajectoryPoint& p = robotTraj.joint_trajectory.points.at(i);
    p.time_from_start = ros::Duration(p.time_from_start.toSec()/scaling_factor);
    for(unsigned int j=0;j<p.velocities.size();j++)
    {
      p.velocities.at(j) *= scaling_factor;
      p.accelerations.at(j) *= scaling_factor*scaling_factor;
    }
  }

  // ROS_INFO_STREAM("Pose target set.");
  auto r = move_group.execute(robotTraj);
  if(r != moveit::planning_interface::MoveItErrorCode::SUCCESS)
    throw std::runtime_error("Trajectory execution for cartesian movement failed. MoveItErrorCode "+std::to_string(r.val));
  // ROS_INFO_STREAM("Moved");
  return 0;
}





int moveToJointPose(moveit::planning_interface::MoveGroupInterface& move_group,
                    std::vector<double> jointPose,
                    double velocity_scaling = 0.1,
                    double acceleration_scaling = 0.1)
{
  moveGroupInt->setMaxAccelerationScalingFactor(acceleration_scaling);
  moveGroupInt->setMaxVelocityScalingFactor(velocity_scaling);
  int r = submitMoveToJointPose(move_group,jointPose);
  if(r<0)
    throw std::runtime_error("Failed to submit move to joint pose, error "+std::to_string(r));

  r = waitActionCompletion(move_group);
  if(r<0)
    throw std::runtime_error("Failed to execute move to joint pose, error "+std::to_string(r));
  return 0;
}


int moveToEePose(moveit::planning_interface::MoveGroupInterface& move_group,
                  const geometry_msgs::PoseStamped& targetPose,
                  std::string endEffectorLink,
                  bool do_cartesian = false,
                  double velocity_scaling = 0.1,
                  double acceleration_scaling = 0.1)
{

  moveGroupInt->setMaxAccelerationScalingFactor(acceleration_scaling);
  moveGroupInt->setMaxVelocityScalingFactor(velocity_scaling);
  int r;
  if(do_cartesian)
  {
    r = executeMoveToEePoseCartesian(move_group, targetPose, endEffectorLink, velocity_scaling);
    if(r<0)
      throw std::runtime_error("Failed to execute cartesian move to end effectore pose, error "+std::to_string(r));
    return 0;
  }
  else
  {
    r = submitMoveToEePose(move_group, targetPose, endEffectorLink);
    if(r<0)
      throw std::runtime_error("Failed to submit move to end effectore pose, error "+std::to_string(r));

    r = waitActionCompletion(move_group);
    if(r<0)
      throw std::runtime_error("Failed to execute move to end effector pose, error "+std::to_string(r));
    return 0;
  }
}

void moveToEePoseActionCallback(const lr_gym_utils::MoveToEePoseGoalConstPtr &goal)
{
  try
  {
    moveToEePose(*moveGroupInt,goal->pose,goal->end_effector_link, goal->do_cartesian, goal->velocity_scaling, goal->acceleration_scaling);
  }
  catch(std::runtime_error& e)
  {
    std::string errorMsg = "moveToEePose failed with error "+std::string(e.what());
    ROS_ERROR_STREAM(errorMsg);
    lr_gym_utils::MoveToEePoseResult result;
    result.succeded = false;
    result.error_message = errorMsg;
    moveToEePoseActionServer->setAborted(result);
    return;
  }

  lr_gym_utils::MoveToEePoseResult result; 
  result.succeded = true;
  result.error_message = "No error";
  moveToEePoseActionServer->setSucceeded(result);

}


void moveToJointPoseActionCallback(const lr_gym_utils::MoveToJointPoseGoalConstPtr &goal)
{
  try
  {
    moveToJointPose(*moveGroupInt,goal->pose, goal->velocity_scaling, goal->acceleration_scaling);
  }
  catch(std::runtime_error& e)
  {
    std::string errorMsg = "moveToJointPose failed with error "+std::string(e.what());
    ROS_ERROR_STREAM(errorMsg);
    lr_gym_utils::MoveToJointPoseResult result;
    result.succeded = false;
    result.error_message = errorMsg;
    moveToJointPoseActionServer->setAborted(result);
    return;
  }

  lr_gym_utils::MoveToJointPoseResult result;
  result.succeded = true;
  result.error_message = "No error";
  moveToJointPoseActionServer->setSucceeded(result);

}


bool getJointStateServiceCallback(lr_gym_utils::GetJointState::Request& req, lr_gym_utils::GetJointState::Response& res)
{
  // ROS_INFO("Getting joint state...");
  moveGroupInt->getCurrentState(10)->copyJointGroupPositions(joint_model_group, res.joint_poses);
  // ROS_INFO("Got joint state.");
  return true;
}

bool getEePoseServiceCallback(lr_gym_utils::GetEePose::Request& req, lr_gym_utils::GetEePose::Response& res)
{
  // ROS_INFO("Getting end effector pose...");
  res.pose = moveGroupInt->getCurrentPose(req.end_effector_link_name);
  // ROS_INFO("Got end effector pose...");
  return true;
}


bool addCollisionBoxServiceCallback(lr_gym_utils::AddCollisionBox::Request& req, lr_gym_utils::AddCollisionBox::Response& res)
{
  moveit_msgs::CollisionObject collision_object;
  collision_object.header.frame_id = req.pose.header.frame_id;

  collision_object.id = "box"+std::to_string(collision_objects_ids.size());
  collision_objects_ids.push_back(collision_object.id);

  shape_msgs::SolidPrimitive primitive;
  primitive.type = primitive.BOX;
  primitive.dimensions.resize(3);
  primitive.dimensions[primitive.BOX_X] = req.size.x;
  primitive.dimensions[primitive.BOX_Y] = req.size.y;
  primitive.dimensions[primitive.BOX_Z] = req.size.z;

  collision_object.primitives.push_back(primitive);
  collision_object.primitive_poses.push_back(req.pose.pose);
  collision_object.operation = collision_object.ADD;

  std::vector<moveit_msgs::CollisionObject> collision_objects;
  collision_objects.push_back(collision_object);
  planningSceneInt->applyCollisionObjects(collision_objects);

  if(req.attach)
  {
    moveGroupInt->attachObject(collision_object.id, req.attach_link, req.attach_ignored_links);
    attached_objects_ids.push_back(collision_object.id);
  }

  res.success = true;
  return true;
}


bool clearCollisionObjectsServiceCallback(lr_gym_utils::ClearCollisionObjects::Request& req, lr_gym_utils::ClearCollisionObjects::Response& res)
{
  // ROS_WARN_STREAM("clearCollisionObjectsServiceCallback");
  for(std::string obj_id : attached_objects_ids)
    moveGroupInt->detachObject(obj_id);
  moveGroupInt->detachObject();//Should detache whatever is attached
  attached_objects_ids.clear();

  moveit_msgs::CollisionObject collision_object;
  std::vector<moveit_msgs::CollisionObject> collision_objects;
  for(std::string obj_id : collision_objects_ids)
  {
    moveit_msgs::CollisionObject collision_object;
    collision_object.id = obj_id;
    collision_object.operation = collision_object.REMOVE;
    collision_objects.push_back(collision_object);
  }
  planningSceneInt->applyCollisionObjects(collision_objects);

  res.objects_count = collision_objects.size();
  collision_objects.clear();

  std::vector<std::string> knownObjs = planningSceneInt->getKnownObjectNames();
  for(std::string obj_id : knownObjs)
  {
    ROS_WARN_STREAM("Found unknown collision object "<<obj_id<<" detaching and removing");
    moveGroupInt->detachObject(obj_id);
    moveit_msgs::CollisionObject collision_object;
    collision_object.id = obj_id;
    collision_object.operation = collision_object.REMOVE;
    collision_objects.push_back(collision_object);
  }
  planningSceneInt->applyCollisionObjects(collision_objects);
  res.objects_count += collision_objects.size();

  //Also, just guess a few of them, as some objects sometimes still persist (Especially attached ones)
  // for(int i=0; i<100; i++)
  // {
  //   std::string obj_id = "box"+std::to_string(i);
  //   moveGroupInt->detachObject(obj_id);
  //   moveit_msgs::CollisionObject collision_object;
  //   collision_object.id = obj_id;
  //   collision_object.operation = collision_object.REMOVE;
  //   collision_objects.push_back(collision_object);
  // }
  // planningSceneInt->applyCollisionObjects(collision_objects);
  // res.objects_count += collision_objects.size();


  return true;
}





int main(int argc, char** argv)
{
  ros::init(argc, argv, "move_helper");
  ros::NodeHandle node_handle("~");

  std::string planning_group_name_param_name = "planning_group_name";
  node_handle.param<std::string>(planning_group_name_param_name, planning_group_name, "panda_arm");
  ROS_INFO_STREAM("Will use planning group "<<planning_group_name);
  std::string default_ee_link_param_name = "default_ee_link";
  node_handle.param<std::string>(default_ee_link_param_name, defaultEeLink, "panda_link8");
  ROS_INFO_STREAM("Will use default ee link "<<defaultEeLink);
  ros::AsyncSpinner spinner(2);
  spinner.start();

  ROS_INFO("Creating MoveGroupInterface...");
  moveGroupInt = std::make_shared<moveit::planning_interface::MoveGroupInterface>(planning_group_name,std::shared_ptr<tf2_ros::Buffer>(),ros::WallDuration(30));
  planningSceneInt = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();
  ROS_INFO("MoveGroupInterface created.");
  joint_model_group = moveGroupInt->getCurrentState()->getJointModelGroup(planning_group_name);

  moveGroupInt->setPlanningTime(0.15);

  tfBuffer = std::make_shared<tf2_ros::Buffer>();
  tf2_ros::TransformListener tfListener(*tfBuffer);


  moveToEePoseActionServer = std::make_shared<actionlib::SimpleActionServer<lr_gym_utils::MoveToEePoseAction>>(node_handle,
                                                                                                                  "move_to_ee_pose",
                                                                                                                  moveToEePoseActionCallback,
                                                                                                                  false);
  moveToEePoseActionServer->start();

  moveToJointPoseActionServer = std::make_shared<actionlib::SimpleActionServer<lr_gym_utils::MoveToJointPoseAction>>(node_handle,
                                                                                                                        "move_to_joint_pose",
                                                                                                                        moveToJointPoseActionCallback,
                                                                                                                        false);
  moveToJointPoseActionServer->start();

  ros::ServiceServer service = node_handle.advertiseService("get_joint_state", getJointStateServiceCallback);
  ros::ServiceServer getEePoseService = node_handle.advertiseService("get_ee_pose", getEePoseServiceCallback);
  ros::ServiceServer addCollisionBoxService = node_handle.advertiseService("add_collision_box", addCollisionBoxServiceCallback);
  ros::ServiceServer clearCollisionObjectsService = node_handle.advertiseService("clear_collision_objects", clearCollisionObjectsServiceCallback);

  ROS_INFO("Action and service servers started");
  ros::waitForShutdown();
  return 0;
}
