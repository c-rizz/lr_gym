#include <gazebo/gazebo.hh>
#include "gazebo/physics/physics.hh"
#include "gazebo/sensors/sensors.hh"
#include "gazebo/rendering/rendering.hh"
#include "gazebo/common/common.hh"

#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <sensor_msgs/fill_image.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include "gazebo_gym_env_plugin/StepSimulation.h"
#include "gazebo_gym_env_plugin/RenderCameras.h"
#include "gazebo_gym_env_plugin/JointInfo.h"
#include "gazebo_gym_env_plugin/JointEffortRequest.h"
#include "gazebo_gym_env_plugin/JointInfoArray.h"
#include "gazebo_gym_env_plugin/GetJointsInfoAction.h"

#include <boost/algorithm/string.hpp>
#include <thread>
#include <mutex>
#include <chrono>
#include <functional>
#include <string>

#include "utils.hpp"

namespace gazebo
{
  /**
   * Gazebo plugin that provides methods necessary for correctly implementing an
   * OpenAI-gym environment.
   */
  class GazeboGymEnvPlugin : public WorldPlugin
  {

  private:

    /**
     * Helper class that wraps a Gazebo camera
     */
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
    event::ConnectionPtr applyJointEffortsCallback;
    std::shared_ptr<std::thread> callbacksThread;
    ros::CallbackQueue callbacksQueue;
    physics::WorldPtr world;
    long stepCounter = 0;
    std::vector<std::shared_ptr<GymCamera>> gymCameras;
    std::queue<std::function<void()>> renderTasksQueue;
    std::mutex renderTasksQueueMutex;


    ros::ServiceServer stepService;
    const std::string stepServiceName = "step";

    ros::ServiceServer renderService;
    const std::string renderServiceName = "render";

    const std::string getJointsInfoActionName = "get_joints_info";

    bool keepServingCallbacks = true;




    AverageKeeper avgRenderThreadDelay;
    AverageKeeper avgRenderTime;
    AverageKeeper avgTotalRenderTime;
    AverageKeeper avgFillTime;
    AverageKeeper avgRenderRequestDelay;
    AverageKeeper avgStepRequestDelay;
    AverageKeeper avgSteppingTime;




    std::vector<gazebo_gym_env_plugin::JointEffortRequest> requestedJointEfforts;
    std::timed_mutex requestedJointEfforts_mutex;

  public:

    virtual ~GazeboGymEnvPlugin()
    {
      //std::cout<<"Destructor!!"<<std::endl;
      keepServingCallbacks = false;
      callbacksThread->join();
    }





    /**
     * Loads the plugin setting up the necessary things
     * @param _parent         [description]
     * @param sdf::ElementPtr [description]
     */
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
      applyJointEffortsCallback = gazebo::event::Events::ConnectWorldUpdateBegin(boost::bind(&GazeboGymEnvPlugin::applyRequestedJointEfforts,this));

      this->nodeHandle = std::make_shared<ros::NodeHandle>("~/gym_env_interface");
      //ROS_INFO("Got node handle");

      this->nodeHandle->setCallbackQueue( &callbacksQueue);


      //Start thread that will handle the service calls
      callbacksThread = std::make_shared<std::thread>(&GazeboGymEnvPlugin::callbacksThreadMain, this);


      stepService = nodeHandle->advertiseService(stepServiceName, &GazeboGymEnvPlugin::stepServiceCallback,this);
      ROS_INFO_STREAM("Advertised service "<<stepServiceName);

      renderService = nodeHandle->advertiseService(renderServiceName, &GazeboGymEnvPlugin::renderServiceCallback,this);
      ROS_INFO_STREAM("Advertised service "<<renderServiceName);


      //world->Physics()->SetSeed(20200413);
    }










  private:

    /**
     * Renders the required cameras by submitting a task to the main Gazebo thread
     * @param  cameras Cameras to be rendered
     * @return         True if the task was executed correctly, false if it timed out
     */
    bool renderCameras(std::vector<std::shared_ptr<GymCamera>> cameras)
    {
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
    bool runOnRenderThreadSync(unsigned int timeoutMillis, std::function<void()> task)
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

    /**
     * Renders the requested cameras
     * @param cameras         Names of the cameras to be rendered
     * @param renderedCameras The renderings are returned in this variable
     */
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

      bool ret = renderCameras(requestedCameras);//renders the cameras on the rendering thread
      if(!ret)
      {
        ROS_WARN("GazeboGymEnvPlugin: Failed to render cameras");
        renderedCameras.success=false;
        renderedCameras.error_message="Renderer task timed out";
        return;
      }

      for(std::shared_ptr<GymCamera> cam : requestedCameras)
          cam->lastRenderedStep = stepCounter;
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


    /**
     * Check if the specified joint exists
     * @param  jointId Joint to check
     * @return         true if it exists
     */
    bool doesJointExist(const gazebo_gym_env_plugin::JointId& jointId)
    {
        gazebo::physics::ModelPtr model = world->ModelByName(jointId.model_name);
        if (!model)
          return false;
        gazebo::physics::JointPtr joint = model->GetJoint(jointId.joint_name);
        if (!joint)
          return false;
        return true;
    }


    /**
     * Gets Position and speed of a joint
     * @param  jointId   The joint to get the information for
     * @param  ret       The result is returned here
     * @return           Positive in case of success, negative in case of error
     */
    int getJointInfo(const gazebo_gym_env_plugin::JointId& jointId, gazebo_gym_env_plugin::JointInfo& ret)
    {

      gazebo::physics::ModelPtr model = world->ModelByName(jointId.model_name);
      if (!model)
        return -1;
      gazebo::physics::JointPtr joint = model->GetJoint(jointId.joint_name);
      if (!joint)
        return -2;

      ret.joint_id = jointId;
      ret.position.clear();
      ret.position.push_back(joint->Position(0));
      ret.rate.clear();
      ret.rate.push_back(joint->GetVelocity(0));

      return 0;
    }

    /**
     * Gets position and speed of a set of joints
     * @param  jointIds  The joints to get the information for
     * @param  ret       The result is returned here (the returned joints are n the same order as in jointIds)
     * @return           Positive in case of success, negative in case of error
     */
    void getJointsInfo(std::vector<gazebo_gym_env_plugin::JointId> jointIds, gazebo_gym_env_plugin::JointsInfoResponse& ret)
    {
      ret.error_message = "";
      for(const gazebo_gym_env_plugin::JointId& jointId : jointIds)
      {
        ROS_INFO_STREAM("Getting joint info for "<<jointId.model_name<<"."<<jointId.joint_name);
        gazebo_gym_env_plugin::JointInfo jointInfo;
        int r = getJointInfo(jointId, jointInfo);
        if(r<0)
        {
          ret.success=false;
          ret.error_message = ret.error_message + "Could not get info for joint " + jointId.model_name + "." + jointId.joint_name+". ";
          ROS_WARN_STREAM(ret.error_message);
        }
        ret.joints_info.push_back(jointInfo);
      }
      ret.success=true;
      ret.error_message="No Error";
    }


    /**
     * Set the effort to be applied on a joint in the next timestep
     * @param  jointId Identifier for the joint
     * @param  effort  Effort to be applied (force or torque depending on the joint type)
     * @return         0 if successfult, negative otherwise
     */
    int setJointEffort(const gazebo_gym_env_plugin::JointId& jointId, double effort)
    {
      gazebo::physics::ModelPtr model = world->ModelByName(jointId.model_name);
      if (!model)
        return -1;
      gazebo::physics::JointPtr joint = model->GetJoint(jointId.joint_name);
      if (!joint)
        return -2;

      joint->SetForce(0,effort); //TODO: do something for joints with more than 1 DOF
      return 0;
    }

    /**
     * Applies the efforts requested by requestJointEffort()
     * @return 0 if successfult, negative if any joint effort request failed
     */
    int applyRequestedJointEfforts()
    {
      std::unique_lock<std::timed_mutex> lk(requestedJointEfforts_mutex,std::chrono::seconds(5));
      if(!lk)
      {
        ROS_ERROR_STREAM("Failed to acquire mutex in "<<__func__<<", aborting");
        return -1;
      }
      //ROS_INFO("Appling efforts");
      int ret = 0;
      for(const gazebo_gym_env_plugin::JointEffortRequest& jer : requestedJointEfforts)
      {
        int r = setJointEffort(jer.joint_id,jer.effort);
        if(r<0)
        {
          ROS_ERROR_STREAM("Failed to apply effort to joint "<<jer.joint_id.model_name<<"."<<jer.joint_id.joint_name);
          ret = -2;
        }
      }
      return ret;
    }

    /**
     * Request a joint effort to be applied on the next timestep
     * @param  request Joint effort request
     * @return         0 if successful
     */
    int requestJointEffort(const gazebo_gym_env_plugin::JointEffortRequest& request)
    {
      std::unique_lock<std::timed_mutex> lk(requestedJointEfforts_mutex,std::chrono::seconds(5));
      if(!lk)
      {
        ROS_ERROR_STREAM("Failed to acquire mutex in "<<__func__<<", aborting");
        return -1;
      }
      requestedJointEfforts.push_back(request);
      return 0;
    }

    /**
     * Clear requests made via requestJointEffort()
     * @return         0 if successful
     */
    int clearRequestedJointEfforts()
    {
      std::unique_lock<std::timed_mutex> lk(requestedJointEfforts_mutex,std::chrono::seconds(5));
      if(!lk)
      {
        ROS_ERROR_STREAM("Failed to acquire mutex in "<<__func__<<", aborting");
        return -1;
      }
      requestedJointEfforts.clear();
      return 0;
    }






















    /**
     * Executed as a thread to handle the ROS service calls
     */
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

    /**
     * Handles a call from the step ROS service
     * @param  req [description]
     * @param  res [description]
     * @return     [description]
     */
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



      res.success = false;
      res.iterations_done = 0;
      res.step_duration_done_secs = 0;
      for(const gazebo_gym_env_plugin::JointEffortRequest& jer : req.joint_effort_requests)
      {
        if(!doesJointExist(jer.joint_id))
        {
          res.error_message = "Requested effort for non-existing joint "+jer.joint_id.model_name+"."+jer.joint_id.joint_name+", aborting step";
          res.response_time = ros::WallTime::now().toSec();
          ROS_WARN_STREAM(res.error_message);
          return true;//must return false only if we cannot send a response
        }
      }
      for(const gazebo_gym_env_plugin::JointId& jid : req.requested_joints)
      {
        if(!doesJointExist(jid))
        {
          res.error_message = "Requested state for non-existing joint "+jid.model_name+"."+jid.joint_name+", aborting step";
          res.response_time = ros::WallTime::now().toSec();
          ROS_WARN_STREAM(res.error_message);
          return true;//must return false only if we cannot send a response
        }
      }


      int requestedIterations = -1;
      if(req.step_duration_secs!=0)
        requestedIterations = req.step_duration_secs/world->Physics()->GetMaxStepSize();
      else
        requestedIterations = req.iterations;

      common::Time startTime = world->SimTime();


      for(const gazebo_gym_env_plugin::JointEffortRequest& jer : req.joint_effort_requests)
      {
        requestJointEffort(jer);
      }



      int iterationsBefore = world->Iterations();
      //ROS_INFO("Stepping simulation...");
      avgSteppingTime.onTaskStart();
      world->Step(requestedIterations);
      avgSteppingTime.onTaskEnd();


      clearRequestedJointEfforts();




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

      if(req.requested_joints.size()>0)
        getJointsInfo(req.requested_joints,res.joints_info);

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


    /**
     * Handles a ROS render service call
     * @param  req [description]
     * @param  res [description]
     * @return     [description]
     */
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
