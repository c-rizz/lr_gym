# True if all the requested rendering were done, false otherwise
bool success

# Rendered images. The order is the same as the one in the cameras field of the request
sensor_msgs/Image[] images

# Cameras information. The order is the same as the one in the cameras field of the request
sensor_msgs/CameraInfo[] camera_infos

# Names of the cameras, in the same order as images and camera_infos
string[] camera_names

# Error message in case of error
string error_message
