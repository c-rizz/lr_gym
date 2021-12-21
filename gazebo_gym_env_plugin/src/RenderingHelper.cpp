#include <sensor_msgs/fill_image.h>
#include <boost/algorithm/string.hpp>
#include "RenderingHelper.hpp"


RenderingHelper::GymCamera::GymCamera(std::shared_ptr<gazebo::sensors::CameraSensor> sensor, std::string rosTfFrame_id)
{
  this->sensor = sensor;
  this->rosTfFrame_id = rosTfFrame_id;
  camera_info = computeCameraInfo();
}

sensor_msgs::CameraInfo RenderingHelper::GymCamera::computeCameraInfo()
{
  unsigned int width  = sensor->ImageWidth();
  unsigned int height = sensor->ImageHeight();

  double cx = (static_cast<double>(height) + 1.0) /2.0;
  double cy = (static_cast<double>(width)  + 1.0) /2.0;


  double fx = width  / (2.0 * tan(sensor->Camera()->HFOV().Radian() / 2.0));
  double fy = height / (2.0 * tan(sensor->Camera()->VFOV().Radian() / 2.0));
  assert(fy!=0);


  double k1 = 0;
  double k2 = 0;
  double k3 = 0;
  double t1 = 0;
  double t2 = 0;
  if(sensor->Camera()->LensDistortion())
  {
    sensor->Camera()->LensDistortion()->SetCrop(true);
    k1 = sensor->Camera()->LensDistortion()->K1();
    k2 = sensor->Camera()->LensDistortion()->K2();
    k3 = sensor->Camera()->LensDistortion()->K3();
    t1 = sensor->Camera()->LensDistortion()->P1();
    t2 = sensor->Camera()->LensDistortion()->P2();
  }

  // fill CameraInfo
  sensor_msgs::CameraInfo camera_info_msg;

  camera_info_msg.header.frame_id = rosTfFrame_id;

  camera_info_msg.height = height;
  camera_info_msg.width  = width;
  // distortion
  camera_info_msg.distortion_model = "plumb_bob";
  camera_info_msg.D.resize(5);


  // D = {k1, k2, t1, t2, k3}, as specified in:
  // - sensor_msgs/CameraInfo: http://docs.ros.org/api/sensor_msgs/html/msg/CameraInfo.html
  // - OpenCV: http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
  camera_info_msg.D = {k1,k2,t1,t2,k3};
  // original camera_ matrix
  camera_info_msg.K = {fx,  0.0, cx,
                       0.0, fy,  cy,
                       0.0, 0.0, 1.0};
  // rectification
  camera_info_msg.R = {1.0, 0.0, 0.0,
                       0.0, 1.0, 0.0,
                       0.0, 0.0, 1.0};
  //This is a monocular camera and there is no rectification, consequently the first 3 columns of P are equal to K, and the last column is (0,0,0)
  camera_info_msg.P = {fx,  0.0, cx,  0.0,
                       0.0, fy,  cy,  0.0,
                       0.0, 0.0, 1.0, 0.0};

  return camera_info_msg;
}

void RenderingHelper::searchCameras()
{
  gazebo::sensors::SensorManager* smanager = gazebo::sensors::SensorManager::Instance();
  std::vector<gazebo::sensors::SensorPtr> sensors = smanager->GetSensors();
  for(gazebo::sensors::SensorPtr sp : sensors)
  {
    if(boost::iequals(sp->Type(),"camera"))
    {
      std::string rosTfFrame_id = "";//TODO: somehow get this
      gymCameras.push_back(std::make_shared<GymCamera>(std::dynamic_pointer_cast<gazebo::sensors::CameraSensor>(sp),rosTfFrame_id));
      ROS_INFO_STREAM("Found camera "<<gymCameras.back()->sensor->Name());
    }
  }
}


RenderingHelper::RenderingHelper(gazebo::physics::WorldPtr world)
{
  this->world = world;
  searchCameras();
  //set renderThreadCallback to be called periodically by the render thread (which is the main thread of gazebo)
  // This ends up being called through the trace:
  //    - Server.cc:Server:: Run()
  //    - sensors::run_once()
  //    - SensorManager::Update()
  //    - ImageSensorContainer::Update()
  //    - event::Events::render()
  renderConnection = gazebo::event::Events::ConnectRender(std::bind(&RenderingHelper::renderThreadCallback, this));
}


/**
 * Renders the required cameras by submitting a task to the main Gazebo thread
 * @param  cameras Cameras to be rendered
 * @return         True if the task was executed correctly, false if it timed out
 */
bool RenderingHelper::renderCameras(std::vector<std::shared_ptr<GymCamera>> cameras)
{
  if(cameras.size()==0)
    return true;
  avgRenderThreadDelay.onTaskStart();
  avgTotalRenderTime.onTaskStart();

  bool ret = runOnRenderThreadSync(10000, [this,cameras](){
    avgRenderThreadDelay.onTaskEnd();
    //Is this a good idea?
    //An issue with this approach is that if things go bad this may be executed super late. And if
    // that happens the cameras may not be usable anymore
    //With other approaches we would probably have the same problems
    for(std::shared_ptr<GymCamera> cam : cameras)
    {
      //Is there a way to check that this will not explode?
      //ROS_INFO_STREAM("### Rendering camera "<<cam->sensor->Camera()->Name());
      avgRenderTime.onTaskStart();
      cam->sensor->Camera()->Render(true);
      cam->sensor->Camera()->PostRender();
      avgRenderTime.onTaskEnd();
      //ROS_INFO_STREAM("### Rendered camera "<<cam->sensor->Camera()->Name());


      //ROS_INFO_STREAM("### Filling image for camera '"<<cam->sensor->Name()<<"'");
      avgFillTime.onTaskStart();
      sensor_msgs::fillImage(cam->lastRender,
            getCameraRosEncoding(cam->sensor),
            cam->sensor->ImageHeight(),
            cam->sensor->ImageWidth(),
            getCameraPixelBytes(cam->sensor)*cam->sensor->ImageWidth(),
            cam->sensor->ImageData());
      avgFillTime.onTaskEnd();
      //ROS_INFO_STREAM("### Built message for camera '"<<cam->sensor->Name()<<"'");
    }
  });

  avgTotalRenderTime.onTaskEnd();
  return ret;
}

/**
 * Run the provided task on the main Gazebo thread and wait for its completion
 * @param  timeoutMillis Timeout for the wait of the task completion
 * @param  task          The task to be executed
 * @return               True if the task completed, false if the timeout expired
 */
bool RenderingHelper::runOnRenderThreadSync(unsigned int timeoutMillis, std::function<void()> task)
{
  std::chrono::milliseconds waitTimeout = std::chrono::milliseconds(timeoutMillis) ;
  std::shared_ptr<bool> taskDone = std::make_shared<bool>(false);
  std::shared_ptr<std::mutex> taskDoneMutex = std::make_shared<std::mutex>();
  std::shared_ptr<std::condition_variable> cv = std::make_shared<std::condition_variable>();

  //ROS_INFO_STREAM("Defining task");
  //define task
  auto taskWrapper = [taskDone,taskDoneMutex,cv,task]() {
    task();
    std::unique_lock<std::mutex> lk(*taskDoneMutex);
    *taskDone = true;
    lk.unlock();
    cv->notify_one();
   };
  //ROS_INFO_STREAM("Defined task");

  //submit task
  {
    std::lock_guard<std::mutex> lock(renderTasksQueueMutex);
    renderTasksQueue.push(taskWrapper);
  }
  //ROS_INFO_STREAM("submitted task");

  //wait completion
  std::unique_lock<std::mutex> lk(*taskDoneMutex);
  bool didCompleteTask = cv->wait_for(lk, waitTimeout, [taskDone]{return *taskDone;});
  lk.unlock();
  //ROS_INFO_STREAM("waited task");

  if(!didCompleteTask)
    ROS_WARN("Failed to run render task. Timed out.");
  return didCompleteTask;
}

/**
 * Periodically called by Gazebo on the main thread. Used to process tasks that
 * require rendering.
 */
void RenderingHelper::renderThreadCallback()
{
  bool done = false;
  do
  {
    std::function<void()> taskToRun;
    {
      std::lock_guard<std::mutex> lock(renderTasksQueueMutex);
      if(renderTasksQueue.empty())
      {
        //ROS_INFO_STREAM("## No tasks to do");
        done = true;
        break;
      }
      else
      {
        //ROS_INFO_STREAM("## there are tasks to do");
        done = false;
        taskToRun = renderTasksQueue.front();
        renderTasksQueue.pop();
        //ROS_INFO_STREAM("## got 1 task");
      }
    }
    if(!done)
    {
      //ROS_INFO_STREAM("## running task");
      taskToRun();
      //ROS_INFO_STREAM("## ran task");
    }
  }while(done);
}

/**
 * Renders the requested cameras
 * @param cameras         Names of the cameras to be rendered
 * @param renderedCameras The renderings are returned in this variable
 */
void RenderingHelper::renderCameras(std::vector<std::string> cameras, gazebo_gym_env_plugin::RenderedCameras& renderedCameras)
{

  std::string availableCamNames = "[";
  for(std::shared_ptr<GymCamera> cam : gymCameras)
    availableCamNames += cam->sensor->Name()+", ";
  availableCamNames += "]";

  //If can't find a camera do a search
  for(std::string reqName : cameras)
  {
    for(int tries=0; tries<2; tries++)
    {
      bool found = false;
      for(std::shared_ptr<GymCamera> cam : gymCameras)
      {
        if(reqName.compare(cam->sensor->Name())==0)
        {
          found = true;
          break;
        }
      }
      if(found)
        break;
      searchCameras();
    }
  }


  ROS_DEBUG("Selecting requested cameras...");
  //Get the cameras we need to use
  std::vector<std::shared_ptr<GymCamera>> requestedCameras;
  for(std::string reqName : cameras)
  {
    bool found = false;
    for(std::shared_ptr<GymCamera> cam : gymCameras)
    {
      if(reqName.compare(cam->sensor->Name())==0)
      {
        requestedCameras.push_back(cam);
        ROS_DEBUG_STREAM("Selecting camera '"<<cam->sensor->Name()<<"'");
        found = true;
        break;
      }
    }
    if(!found)
    {
      ROS_WARN_STREAM("Couldn't find requested camera '"<<reqName<<"' available cameras are: "<<availableCamNames<<std::endl);
    }
  }
  ROS_DEBUG_STREAM("Selected "<<requestedCameras.size()<<" cameras");

  if(requestedCameras.size()>0)
  {
    bool ret = renderCameras(requestedCameras);//renders the cameras on the rendering thread
    if(!ret)
    {
      ROS_WARN("GazeboGymEnvPlugin: Failed to render cameras");
      renderedCameras.success=false;
      renderedCameras.error_message="Renderer task timed out";
      return;
    }
  }
  
  //Fill up the response with the images
  gazebo::common::Time simTime = world->SimTime();
  for(std::shared_ptr<GymCamera> cam  : requestedCameras)
  {
    //ROS_INFO_STREAM("Building message for camera '"<<cam->sensor->Name()<<"'");
    renderedCameras.images.push_back(cam->lastRender);

    renderedCameras.images.back().header.stamp.sec = simTime.sec;
    renderedCameras.images.back().header.stamp.nsec = simTime.nsec;
    renderedCameras.images.back().header.frame_id = cam->rosTfFrame_id;
    //ROS_INFO_STREAM("Built message for camera '"<<cam->sensor->Name()<<"'");
  }


  //ROS_INFO_STREAM("Setting camera infos...");
  //Fill up camera infos
  for(std::shared_ptr<GymCamera> cam : requestedCameras)
    renderedCameras.camera_infos.push_back(cam->camera_info);
  //ROS_INFO_STREAM("Done");

  for(std::shared_ptr<GymCamera> cam : requestedCameras)
    renderedCameras.camera_names.push_back(cam->sensor->Name());

  renderedCameras.success=true;
}



double RenderingHelper::getAvgTotalRenderTime()
{
  return avgTotalRenderTime.getAverage();
}

double RenderingHelper::getAvgFillTime()
{
  return avgFillTime.getAverage();
}

double RenderingHelper::getAvgRenderThreadDelay()
{
  return avgRenderThreadDelay.getAverage();
}

double RenderingHelper::getAvgRenderTime()
{
  return avgRenderTime.getAverage();
}
