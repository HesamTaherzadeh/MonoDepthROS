cmake_minimum_required(VERSION 3.0.2)
project(robot_depth_estimation)

## Compile as C++17, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  cv_bridge
  rospy
  std_msgs
  xacro
)

## Find OpenCV
find_package(OpenCV REQUIRED)

## Find ONNX Runtime
find_package(onnxruntime REQUIRED)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS gazebo_ros rospy std_msgs xacro
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${onnxruntime_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare a C++ executable
add_executable(${PROJECT_NAME}_node src/depth_estimation_node.cpp src/depth_infer.cpp)

## Add dependencies for the executable
add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Link ONNX Runtime library and catkin libraries
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${onnxruntime_LIBRARIES}  
  ${OpenCV_LIBS}
  /usr/local/lib/libonnxruntime.so
)

#############
## Install ##
#############

# Mark executables for installation
install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#############
## Testing ##
#############

# Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_robot_depth_estimation.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

# Add folders to be run by python nosetests
# catkin_add_nosetests(test)
