#ifndef RENDERING_HELPER_HPP_20200528
#define RENDERING_HELPER_HPP_20200528

#include <string>
#include <memory>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include "gazebo/sensors/sensors.hh"
#include "gazebo/rendering/rendering.hh"
#include "gazebo/common/common.hh"
#include "utils.hpp"
#include "gazebo_gym_env_plugin/RenderedCameras.h"


class RenderingHelper
{
private:
    /**
     * Helper class that wraps a Gazebo camera
     */
    class GymCamera
    {
    public:
      std::shared_ptr<gazebo::sensors::CameraSensor> sensor;
      std::string rosTfFrame_id;
      sensor_msgs::CameraInfo camera_info;
      sensor_msgs::Image lastRender;

      GymCamera(std::shared_ptr<gazebo::sensors::CameraSensor> sensor, std::string rosTfFrame_id);
    private:
      sensor_msgs::CameraInfo computeCameraInfo();
    };

    gazebo::physics::WorldPtr world;
    gazebo::event::ConnectionPtr renderConnection;//conected gazebo events
    std::vector<std::shared_ptr<GymCamera>> gymCameras;
    std::queue<std::function<void()>> renderTasksQueue;
    std::mutex renderTasksQueueMutex;
    AverageKeeper avgTotalRenderTime;
    AverageKeeper avgFillTime;
    AverageKeeper avgRenderThreadDelay;
    AverageKeeper avgRenderTime;
public:

  RenderingHelper(gazebo::physics::WorldPtr world);

  /**
   * Renders the required cameras by submitting a task to the main Gazebo thread
   * @param  cameras Cameras to be rendered
   * @return         True if the task was executed correctly, false if it timed out
   */
  bool renderCameras(std::vector<std::shared_ptr<GymCamera>> cameras);

  /**
   * Run the provided task on the main Gazebo thread and wait for its completion
   * @param  timeoutMillis Timeout for the wait of the task completion
   * @param  task          The task to be executed
   * @return               True if the task completed, false if the timeout expired
   */
  bool runOnRenderThreadSync(unsigned int timeoutMillis, std::function<void()> task);

  /**
   * Periodically called by Gazebo on the main thread. Used to process tasks that
   * require rendering.
   */
  void renderThreadCallback();

  /**
   * Renders the requested cameras
   * @param cameras         Names of the cameras to be rendered
   * @param renderedCameras The renderings are returned in this variable
   */
  void renderCameras(std::vector<std::string> cameras, gazebo_gym_env_plugin::RenderedCameras& renderedCameras);

  double getAvgTotalRenderTime();
  double getAvgFillTime();
  double getAvgRenderThreadDelay();
  double getAvgRenderTime();
};


#endif
