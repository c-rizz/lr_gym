#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <gazebo/gazebo.hh>
#include "gazebo/sensors/sensors.hh"
#include "gazebo/rendering/rendering.hh"


class AverageKeeper
{
private:
  std::vector<double> buffer;
  unsigned int pos = 0;
  unsigned int bufSize = 100;
  double avg = 0;
  std::chrono::steady_clock::time_point taskStartTime;

public:

  double addValue(double newVal);

  void onTaskStart();

  void onTaskEnd();

  double getAverage();
};


std::string getCameraRosEncoding(std::shared_ptr<gazebo::sensors::CameraSensor> sensor);

unsigned int getCameraRosSkip(std::shared_ptr<gazebo::sensors::CameraSensor> sensor);
