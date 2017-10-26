cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(feature_extractor)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

## catkin_make -DCMAKE_BUILD_TYPE=Debug

add_executable (FE	FE.cpp	cylinder_segmentation.cpp	read_matrices_pose.cpp	read_transformations.cpp)

target_link_libraries (FE ${PCL_LIBRARIES})

#add_executable (cylinder_segmentation cylinder_segmentation.cpp)

#target_link_libraries (cylinder_segmentation ${PCL_LIBRARIES})

#add_executable (read_matrices_pose read_matrices_pose.cpp)

#target_link_libraries (read_matrices_pose ${PCL_LIBRARIES})

#add_executable (read_transformations read_transformations.cpp)

#target_link_libraries (read_transformations ${PCL_LIBRARIES})

#add_executable (read_velo_to_cam read_velo_to_cam.cpp)

#target_link_libraries (read_velo_to_cam ${PCL_LIBRARIES})


