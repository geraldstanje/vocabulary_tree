cmake_minimum_required(VERSION 2.8.3)
project(vocabulary_tree)

find_package(catkin REQUIRED COMPONENTS
    roscpp
)
find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(include 
    ${catkin_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
add_definitions("-msse3 -mfpmath=sse")

catkin_package(
    LIBRARIES vocabulary_tree
    INCLUDE_DIRS include
    DEPENDS Eigen OpenCV
)

add_library(vocabulary_tree
    src/database.cpp
    src/generic_tree.cpp
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h")

install(TARGETS vocabulary_tree DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
