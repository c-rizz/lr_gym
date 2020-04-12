#include <ros/ros.h>
#include <gazebo/gazebo.hh>
#include "gazebo/sensors/sensors.hh"
#include "gazebo/rendering/rendering.hh"
#include <sensor_msgs/image_encodings.h>
#include "utils.hpp"

using namespace gazebo;



double AverageKeeper::addValue(double newVal)
{
  if(buffer.size()<bufSize)
    buffer.push_back(newVal);
  else
    buffer[pos%bufSize] = newVal;
  pos++;

  double sum = 0;
  for(double v : buffer)
    sum+=v;
  avg = sum/buffer.size();
  return avg;
}

void AverageKeeper::onTaskStart()
{
  taskStartTime = std::chrono::steady_clock::now();
}

void AverageKeeper::onTaskEnd()
{
  auto taskEndTime = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration = taskEndTime-taskStartTime;
  addValue(duration.count());
}

double AverageKeeper::getAverage()
{
  return avg;
}






std::string getCameraRosEncoding(std::shared_ptr<sensors::CameraSensor> sensor)
{
  std::string sensorFormat = sensor->Camera()->ImageFormat();
  std::string ret;
  //Thanks to gazebo_ros_pkgs/gazebo_plugins/src/gazebo_ros_camera_utils.cpp
  if (sensorFormat == "L8" || sensorFormat == "L_INT8")
    ret = sensor_msgs::image_encodings::MONO8;
  else if (sensorFormat == "L16" || sensorFormat == "L_INT16")
    ret = sensor_msgs::image_encodings::MONO16;
  else if (sensorFormat == "R8G8B8" || sensorFormat == "RGB_INT8")
    ret = sensor_msgs::image_encodings::RGB8;
  else if (sensorFormat == "B8G8R8" || sensorFormat == "BGR_INT8")
    ret = sensor_msgs::image_encodings::BGR8;
  else if (sensorFormat == "R16G16B16" ||  sensorFormat == "RGB_INT16")
    ret = sensor_msgs::image_encodings::RGB16;
  else if (sensorFormat == "BAYER_RGGB8")
  {
    ROS_INFO_STREAM("gazebo_gym_env: bayer simulation maybe computationally expensive.");
    ret = sensor_msgs::image_encodings::BAYER_RGGB8;
  }
  else if (sensorFormat == "BAYER_BGGR8")
  {
    ROS_INFO_STREAM("gazebo_gym_env: bayer simulation maybe computationally expensive.");
    ret = sensor_msgs::image_encodings::BAYER_BGGR8;
  }
  else if (sensorFormat == "BAYER_GBRG8")
  {
    ROS_INFO_STREAM("gazebo_gym_env: bayer simulation maybe computationally expensive.");
    ret = sensor_msgs::image_encodings::BAYER_GBRG8;
  }
  else if (sensorFormat == "BAYER_GRBG8")
  {
    ROS_INFO_STREAM("gazebo_gym_env: bayer simulation maybe computationally expensive.");
    ret = sensor_msgs::image_encodings::BAYER_GRBG8;
  }
  else
  {
    ROS_ERROR_STREAM("gazebo_gym_env: Unsupported Gazebo ImageFormat "<<sensorFormat<<" on sensor "<<sensor->Name());
    ret = sensor_msgs::image_encodings::BGR8;
  }

  //ROS_INFO_STREAM("Got encoding: "<<ret);
  return ret;
}


unsigned int getCameraRosSkip(std::shared_ptr<sensors::CameraSensor> sensor)
{
  std::string sensorFormat = sensor->Camera()->ImageFormat();
  unsigned int ret;

  //Thanks to gazebo_ros_pkgs/gazebo_plugins/src/gazebo_ros_camera_utils.cpp
  if (sensorFormat == "L8" || sensorFormat == "L_INT8")
    ret = 1;
  else if (sensorFormat == "L16" || sensorFormat == "L_INT16")
    ret = 2;
  else if (sensorFormat == "R8G8B8" || sensorFormat == "RGB_INT8")
    ret = 3;
  else if (sensorFormat == "B8G8R8" || sensorFormat == "BGR_INT8")
    ret = 3;
  else if (sensorFormat == "R16G16B16" ||  sensorFormat == "RGB_INT16")
    ret = 6;
  else if (sensorFormat == "BAYER_RGGB8")
  {
    ROS_INFO_STREAM("gazebo_gym_env: bayer simulation maybe computationally expensive.");
    ret = 1;
  }
  else if (sensorFormat == "BAYER_BGGR8")
  {
    ROS_INFO_STREAM("gazebo_gym_env: bayer simulation maybe computationally expensive.");
    ret = 1;
  }
  else if (sensorFormat == "BAYER_GBRG8")
  {
    ROS_INFO_STREAM("gazebo_gym_env: bayer simulation maybe computationally expensive.");
    ret = 1;
  }
  else if (sensorFormat == "BAYER_GRBG8")
  {
    ROS_INFO_STREAM("gazebo_gym_env: bayer simulation maybe computationally expensive.");
    ret = 1;
  }
  else
  {
    ROS_ERROR_STREAM("gazebo_gym_env: Unsupported Gazebo ImageFormat "<<sensorFormat<<" on sensor "<<sensor->Name());
    ret = 3;
  }

  //ROS_INFO_STREAM("Got step: "<<ret);
  return ret;
}
