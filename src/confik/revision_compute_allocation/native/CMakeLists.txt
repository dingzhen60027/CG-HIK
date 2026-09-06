cmake_minimum_required(VERSION 3.16)
project(revision_trac_adapter LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(rclcpp REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(urdf REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(NLopt REQUIRED)
add_library(revision_trac SHARED
  trac_adapter.cpp
  ${TRAC_SOURCE}/trac_ik_lib/src/trac_ik.cpp
  ${TRAC_SOURCE}/trac_ik_lib/src/nlopt_ik.cpp
  ${TRAC_SOURCE}/trac_ik_lib/src/kdl_tl.cpp)
target_include_directories(revision_trac PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compat ${TRAC_SOURCE}/trac_ik_lib/include ${NLOPT_INCLUDE_DIRS})
target_link_libraries(revision_trac PRIVATE rclcpp::rclcpp kdl_parser::kdl_parser urdf::urdf Eigen3::Eigen ${NLOPT_LIBRARIES})
