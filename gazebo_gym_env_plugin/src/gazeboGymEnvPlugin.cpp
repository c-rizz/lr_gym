#include <gazebo/gazebo.hh>
#include "gazebo/physics/physics.hh"
#include "gazebo/sensors/sensors.hh"
#include "gazebo/rendering/rendering.hh"
#include "gazebo/common/common.hh"
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include "gazebo_gym_env_plugin/StepSimulation.h"
#include "gazebo_gym_env_plugin/RenderCameras.h"
#include <boost/algorithm/string.hpp>
#include <thread>
#include <sensor_msgs/fill_image.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <mutex>
#include <chrono>
#include <functional>
#include "utils.hpp"

namespace gazebo
{
  class GazeboGymEnvPlugin : public WorldPlugin
  {

  private:

    class GymCamera
    {
    public:
      std::shared_ptr<sensors::CameraSensor> sensor;
      long lastRenderedStep;
      std::string rosTfFrame_id;
      sensor_msgs::CameraInfo camera_info;
      sensor_msgs::Image lastRender;

      GymCamera(std::shared_ptr<sensors::CameraSensor> sensor, std::string rosTfFrame_id)
      {
        this->sensor = sensor;
        this->lastRenderedStep = -1;
        this->rosTfFrame_id = rosTfFrame_id;
        camera_info = computeCameraInfo();
      }
    private:
      sensor_msgs::CameraInfo computeCameraInfo()
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
    };

    std::shared_ptr<ros::NodeHandle> nodeHandle;
    event::ConnectionPtr renderConnection;//conected gazebo events
    std::shared_ptr<std::thread> callbacksThread;
    ros::CallbackQueue callbacksQueue;
    physics::WorldPtr world;
    long stepCounter = 0;
    std::vector<std::shared_ptr<GymCamera>> gymCameras;
    std::queue<std::function<void()>> renderTasksQueue;
    std::mutex renderTasksQueueMutex;
    std::chrono::milliseconds waitTimeout = std::chrono::milliseconds(10000) ;


    ros::ServiceServer stepService;
    const std::string stepServiceName = "step";

    ros::ServiceServer renderService;
    const std::string renderServiceName = "render";

    bool keepServingCallbacks = true;




    AverageKeeper avgRenderThreadDelay;
    AverageKeeper avgRenderTime;
    AverageKeeper avgTotalRenderTime;
    AverageKeeper avgFillTime;
    AverageKeeper avgRenderRequestDelay;
    AverageKeeper avgStepRequestDelay;
    AverageKeeper avgSteppingTime;

  public:
    /////////////////////////////////////////////
    /// \brief Destructor
    virtual ~GazeboGymEnvPlugin()
    {
      //std::cout<<"Destructor!!"<<std::endl;
      keepServingCallbacks = false;
      callbacksThread->join();
    }

    /////////////////////////////////////////////
    /// \brief Called after the plugin has been constructed.
    void Load(physics::WorldPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
      if (!ros::isInitialized())
      {
        ROS_FATAL_STREAM_NAMED("GazeboGymEnvPlugin", "A ROS node for Gazebo has not been initialized, unable to load plugin. "
          << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
        return;
      }

      world = _parent;

      sensors::SensorManager* smanager = sensors::SensorManager::Instance();
      std::vector<sensors::SensorPtr> sensors = smanager->GetSensors();
      for(sensors::SensorPtr sp : sensors)
      {
        if(boost::iequals(sp->Type(),"camera"))
        {
          std::string rosTfFrame_id = "";//TODO: somehow get this
          gymCameras.push_back(std::make_shared<GymCamera>(std::dynamic_pointer_cast<sensors::CameraSensor>(sp),rosTfFrame_id));
        }
      }

      //set renderThreadCallback to be called periodically by the render thread (which is the main thread of gazebo)
      // This ends up being called through the trace:
      //    - Server.cc:Server:: Run()
      //    - sensors::run_once()
      //    - SensorManager::Update()
      //    - ImageSensorContainer::Update()
      //    - event::Events::render()
      renderConnection = event::Events::ConnectRender(std::bind(&GazeboGymEnvPlugin::renderThreadCallback, this));


      this->nodeHandle = std::make_shared<ros::NodeHandle>("~/gym_env_interface");
      ROS_INFO("Got node handle");



      callbacksThread = std::make_shared<std::thread>(&GazeboGymEnvPlugin::callbacksThreadMain, this);

      ros::AdvertiseServiceOptions step_service_aso = ros::AdvertiseServiceOptions::create<gazebo_gym_env_plugin::StepSimulation>(
                                                                    stepServiceName,
                                                                    boost::bind(&GazeboGymEnvPlugin::stepServiceCallback,this,_1,_2),
                                                                    ros::VoidPtr(), &callbacksQueue);
      stepService = nodeHandle->advertiseService(step_service_aso);


      ros::AdvertiseServiceOptions render_service_aso = ros::AdvertiseServiceOptions::create<gazebo_gym_env_plugin::RenderCameras>(
                                                                    renderServiceName,
                                                                    boost::bind(&GazeboGymEnvPlugin::renderServiceCallback,this,_1,_2),
                                                                    ros::VoidPtr(), &callbacksQueue);
      renderService = nodeHandle->advertiseService(render_service_aso);

      ROS_INFO("Advertised service ");
    }

  private:


    bool renderCameras(std::vector<std::shared_ptr<GymCamera>> cameras)
    {
      avgRenderThreadDelay.onTaskStart();
      avgTotalRenderTime.onTaskStart();

      bool ret = runOnRenderThreadSync([this,cameras](){
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
                getCameraRosSkip(cam->sensor)*cam->sensor->ImageWidth(),
                cam->sensor->ImageData());
          avgFillTime.onTaskEnd();
          //ROS_INFO_STREAM("### Built message for camera '"<<cam->sensor->Name()<<"'");
        }
      });

      avgTotalRenderTime.onTaskEnd();
      return ret;
    }

    bool runOnRenderThreadSync(std::function<void()> task)
    {
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

    void renderThreadCallback()
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

    void renderCameras(std::vector<std::string> cameras, gazebo_gym_env_plugin::RenderedCameras& renderedCameras)
    {
      ROS_DEBUG("Available Cameras:");
      for(std::shared_ptr<GymCamera> cam : gymCameras)
        ROS_DEBUG_STREAM("  "<<cam->sensor->Name());


      ROS_DEBUG("Selecting requested cameras...");
      //Get the cameras we need to use
      std::vector<std::shared_ptr<GymCamera>> requestedCameras;
      if(cameras.empty())
      {
        requestedCameras = gymCameras;
      }
      else
      {
        for(std::string reqName : cameras)
        {
          for(std::shared_ptr<GymCamera> cam : gymCameras)
          {
            if(reqName.compare(cam->sensor->Name())==0)
            {
              requestedCameras.push_back(cam);
              ROS_DEBUG_STREAM("Selecting camera '"<<cam->sensor->Name()<<"'");
              break;
            }
          }
        }
      }
      ROS_DEBUG_STREAM("Selected "<<requestedCameras.size()<<" cameras");

      //Render cameras that were not rendered since last step end
      std::vector<std::shared_ptr<GymCamera>> camerasToRender;
      for(std::shared_ptr<GymCamera> cam : requestedCameras)
      {
        //ROS_INFO_STREAM("camera '"<<cam->sensor->Name()<<"': lastRenderedStep="<<cam->lastRenderedStep<<", stepCounter ="<<stepCounter);
        if(cam->lastRenderedStep<stepCounter)
        {
          camerasToRender.push_back(cam);
          cam->lastRenderedStep = stepCounter;
        }
      }
      bool ret = renderCameras(camerasToRender);//renders the cameras on the rendering thread
      if(!ret)
      {
        ROS_WARN("GazeboGymEnvPlugin: Failed to render cameras");
        renderedCameras.success=false;
        renderedCameras.error_message="Renderer task timed out";
        return;
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













    void callbacksThreadMain()
    {
      //Initialize rendering engine in this thread (necessary for rendeing the camera)
      //rendering::load();
      //rendering::init();
      static const double timeout = 0.001;
      while(keepServingCallbacks)
      {
        //ROS_INFO("Looping callbacksThreadMain");
        callbacksQueue.callAvailable(ros::WallDuration(timeout));
      }
      //close the rendering engine for this thread
      //rendering::fini();
    }

    bool stepServiceCallback(gazebo_gym_env_plugin::StepSimulation::Request &req, gazebo_gym_env_plugin::StepSimulation::Response &res)
    {
      /*
      By looking at the code in gazebo/physics/World.cc:
        stop makes the simulation loop stop
        pause makes the simulation loop do nothing (but it still loops!)
        world->Running() indicates if the simulation loop is stopped
        world->IsPaused() indicates if the simulation is paused
        Step(int _steps) makes the simulation run even if it is paused, for _steps steps

      So:
        We keep the simulation paused and we make it go forward with Step(int). We cannot suse stop because it
        also stops the sensor updates, which prevents us from using the cameras, we cannot render them ourselves
        because even the Render event gets stopped.
      */
      if(req.iterations !=0 && req.step_duration_secs!=0)
      {
        res.success = false;
        res.error_message = "GazeboGymEnvPlugin: step was requested specifying both iterations and step_duration. Only one can be set at a time. No action taken.";
        ROS_WARN_STREAM(res.error_message.c_str());
        return true;
      }

      if(!world->IsPaused())
      {
        res.success = false;
        res.error_message = "Called step_simulation while the simulation was running. Simulation must be paused. No action taken.";
        ROS_WARN_STREAM(res.error_message.c_str());
        return true;
      }

      double delay_secs = ros::WallTime::now().toSec() - req.request_time;
      avgStepRequestDelay.addValue(delay_secs);
      //ROS_INFO_STREAM("Stepping simulation. Service request delay = "<<delay_secs);

      int requestedIterations = -1;
      if(req.step_duration_secs!=0)
        requestedIterations = req.step_duration_secs/world->Physics()->GetMaxStepSize();
      else
        requestedIterations = req.iterations;

      common::Time startTime = world->SimTime();

      int iterationsBefore = world->Iterations();
      //ROS_INFO("Stepping simulation...");
      avgSteppingTime.onTaskStart();
      world->Step(requestedIterations);
      avgSteppingTime.onTaskEnd();

      common::Time endTime = world->SimTime();

      int iterations_done = world->Iterations() - iterationsBefore;
      res.success = iterations_done == requestedIterations;
      res.error_message = "No error";
      res.iterations_done = iterations_done;
      res.step_duration_done_secs = (endTime-startTime).Double();
      res.response_time = ros::WallTime::now().toSec();

      stepCounter++;

      if(req.render)
      {
        renderCameras(req.cameras,res.render_result);
      }

      //Print timing info
      ROS_INFO_STREAM("-------------------------------------------------");
      ROS_INFO_STREAM("Render request delay:         avg="<<avgRenderRequestDelay.getAverage()*1000<<"ms");
      ROS_INFO_STREAM("Render thread call delay:     avg="<<avgRenderThreadDelay.getAverage()*1000<<"ms");
      ROS_INFO_STREAM("Render duration:              avg="<<avgRenderTime.getAverage()*1000<<"ms");
      ROS_INFO_STREAM("Image fill duration:          avg="<<avgFillTime.getAverage()*1000<<"ms");
      ROS_INFO_STREAM("Total Render duration:        avg="<<avgTotalRenderTime.getAverage()*1000<<"ms");
      ROS_INFO_STREAM("Step request delay:           avg="<<avgStepRequestDelay.getAverage()*1000<<"ms");
      ROS_INFO_STREAM("Step wall duration:           avg="<<avgSteppingTime.getAverage()*1000<<"ms");

      return true;//Must be false only in case we cannot send a response
    }



    bool renderServiceCallback(gazebo_gym_env_plugin::RenderCameras::Request &req, gazebo_gym_env_plugin::RenderCameras::Response &res)
    {
      double delay_secs = ros::WallTime::now().toSec() - req.request_time;
      avgRenderRequestDelay.addValue(delay_secs);
      //ROS_INFO_STREAM("Rendering cameras. Service request delay = "<<delay_secs);

      renderCameras(req.cameras,res.render_result);

      res.response_time = ros::WallTime::now().toSec();
      return true;//Must be false only in case we cannot send a response
    }

  };

  // Register this plugin with the simulator
  GZ_REGISTER_WORLD_PLUGIN(GazeboGymEnvPlugin)
}
