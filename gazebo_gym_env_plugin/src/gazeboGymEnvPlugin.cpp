#include <gazebo/gazebo.hh>
#include "gazebo/physics/physics.hh"
#include "gazebo/common/common.hh"
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include "gazebo_gym_env_plugin/StepSimulation.h"

#include <thread>

namespace gazebo
{
  class GazeboGymEnvPlugin : public WorldPlugin
  {

  private:
    std::shared_ptr<ros::NodeHandle> nodeHandle;
    std::shared_ptr<std::thread> callbacksThread;
    ros::CallbackQueue callbacksQueue;
    physics::WorldPtr world;

    ros::ServiceServer stepService;
    const std::string stepSimulationServiceName = "step_simulation";
    bool keepServingCallbacks = true;

  public:
    /////////////////////////////////////////////
    /// \brief Destructor
    virtual ~GazeboGymEnvPlugin()
    {
      std::cout<<"Destructor!!"<<std::endl;
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

      this->nodeHandle = std::make_shared<ros::NodeHandle>("gym_env_interface");
      ROS_INFO("Got node handle");

      callbacksThread = std::make_shared<std::thread>(&GazeboGymEnvPlugin::callbacksThreadMain, this);

      ros::AdvertiseServiceOptions step_service_aso = ros::AdvertiseServiceOptions::create<gazebo_gym_env_plugin::StepSimulation>(
                                                                    stepSimulationServiceName,
                                                                    boost::bind(&GazeboGymEnvPlugin::stepSimulation,this,_1,_2),
                                                                    ros::VoidPtr(), &callbacksQueue);
      stepService = nodeHandle->advertiseService(step_service_aso);
      ROS_INFO("Advertised service ");
    }

  private:
    void callbacksThreadMain()
    {
      static const double timeout = 0.001;
      while(keepServingCallbacks)
      {
        //ROS_INFO("Looping callbacksThreadMain");
        callbacksQueue.callAvailable(ros::WallDuration(timeout));
      }
    }

    /////////////////////////////////////////////
    // \brief Called once after Load
    void Init()
    {
      std::cout<<"Init!"<<std::endl;
    }

    bool stepSimulation(gazebo_gym_env_plugin::StepSimulation::Request &req, gazebo_gym_env_plugin::StepSimulation::Response &res)
    {
      /*
      By looking at the code in gazebo/physics/World.cc:
        stop makes the simulation loop stop
        pause makes the simulation loop do nothing (but it still loops!)
        world->Running() indicates if the simulation loop is stopped
        world->IsPaused() indicates if the simulation is paused
      */
      if(world->Running())
      {
        ROS_WARN("Called step_simulation while the simulation was running! This is probably a mistake. I will stop the simulation and then do a step");
      }

      int requestedIterations = req.iterations;

      ROS_INFO("Stepping simulation...");
      //Ensure there is no simulation loop currently running
      world->Stop();//I think this also waits for the simulation loop that is currently running to finish

      world->SetPaused(false);//If it were paused the simulation loop would run but it wouldn't do anything

      world->RunBlocking(requestedIterations);

      int iterations_done = world->Iterations();
      res.success = iterations_done == requestedIterations;
      res.iterations_done = iterations_done;
      return true;//Would have to be false if we couldn't send a response
    }

  };

  // Register this plugin with the simulator
  GZ_REGISTER_WORLD_PLUGIN(GazeboGymEnvPlugin)
}
