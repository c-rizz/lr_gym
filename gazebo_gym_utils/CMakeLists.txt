cmake_minimum_required(VERSION 2.8.3)
project(gazebo_gym_utils)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11 -Wall)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  simple_panda_moveit_config
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  moveit_ros_perception
  actionlib_msgs
  rviz_visual_tools
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  tf2_kdl
  tf2_msgs
  tf2_py
  kdl_parser
  kdl_conversions
  geometry_msgs
  controller_interface
  hardware_interface
  pluginlib
  realtime_tools
)

## System dependencies are found with CMake's conventions

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

find_package(orocos_kdl REQUIRED)
find_package(urdfdom_headers REQUIRED)


################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  LinkStates.msg
)

## Generate services in the 'srv' folder
add_service_files(
	FILES
	GetEePose.srv
	GetJointState.srv
)

## Generate actions in the 'action' folder
add_action_files(
	FILES
	MoveToEePose.action
	MoveToJointPose.action
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  actionlib_msgs
  geometry_msgs
)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  LIBRARIES gazebo_gym_utils
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    simple_panda_moveit_config
    moveit_core
    moveit_visual_tools
    moveit_ros_planning_interface
    interactive_markers
    tf2_geometry_msgs
    actionlib_msgs
    geometry_msgs
    rviz_visual_tools
    tf2
    tf2_ros
	  kdl_parser
	  kdl_conversions

  DEPENDS
    EIGEN3
	  orocos_kdl
	  urdfdom_headers
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
#  include
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS} ${orocos_kdl_INCLUDE_DIRS} ${urdfdom_headers_INCLUDE_DIRS}
)

link_directories(${orocos_kdl_LIBRARY_DIRS})


add_executable(move_helper src/move_helper.cpp)
add_dependencies(move_helper ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(move_helper ${catkin_LIBRARIES} ${Boost_LIBRARIES})

add_executable(link_state_publisher src/link_state_publisher.cpp)
add_dependencies(link_state_publisher ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(link_state_publisher ${catkin_LIBRARIES} ${orocos_kdl_LIBRARIES})




# Declare a C++ library
add_library(gazebo_gym_utils
  src/GravityCompensatedEffortController.cpp
)

add_dependencies(gazebo_gym_utils
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(gazebo_gym_utils PUBLIC
  ${catkin_LIBRARIES}
)

target_include_directories(gazebo_gym_utils SYSTEM PUBLIC
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)
target_include_directories(gazebo_gym_utils PUBLIC
  include
)

## Installation
install(TARGETS gazebo_gym_utils
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(FILES gazebo_gym_utils_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
