cmake_minimum_required(VERSION 3.8)
project(depth_inference)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_library(CUDNN_LIBRARY cudnn HINTS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)
find_package(tf2_geometry_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(yaets REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(GTSAM REQUIRED)
find_package(GTSAM_UNSTABLE REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)

if(NOT onnxruntime_FOUND)
  message(FATAL_ERROR "onnxruntime not found. Make sure it's installed and accessible.")
else()
  message(STATUS "onnxruntime found at: ${onnxruntime_DIR}")
endif()

# Include directories
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIRS}
)

################KIMERA
include_directories(/home/hesam/Desktop/Repos/Kimera-VIO/include)
link_directories(/home/hesam/Desktop/Repos/Kimera-VIO/build/lib)
################

# Add executables
add_executable(slam_node src/depth_node.cpp src/context.cpp src/models.cpp)
add_executable(utils_node src/utils.cpp)
add_executable(optimizer_node  src/optimizer_node.cpp src/optimizer.cpp) 

target_link_libraries(slam_node
  onnxruntime
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${CUDNN_LIBRARY}
  ${PCL_LIBRARIES}
  /home/hesam/Desktop/Repos/Kimera-VIO/build/libkimera_vio.so
  glog
  gtsam
  gtsam_unstable
)

target_link_libraries(optimizer_node
  gtsam
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

ament_target_dependencies(slam_node
  yaets
  tf2_ros
  rclcpp
  pcl_conversions
  cv_bridge
  nav_msgs
  std_msgs
  sensor_msgs
  tf2_geometry_msgs
  OpenCV
  CUDA
  PCL
)

ament_target_dependencies(utils_node
  rclcpp
  sensor_msgs
)

# Specify dependencies for optimizer_node
ament_target_dependencies(optimizer_node
  rclcpp
  nav_msgs
  sensor_msgs
  tf2_ros
  tf2_geometry_msgs
  GTSAM         
  OpenCV
  PCL
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Install targets
install(TARGETS slam_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS utils_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS optimizer_node  # Install optimizer_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)

install(DIRECTORY cfg/
  DESTINATION share/${PROJECT_NAME}/cfg
)

install(
  PROGRAMS
  scripts/odometry.py
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
