cmake_minimum_required(VERSION 2.8)
project(Robot_Calibration)
find_package(OpenCV REQUIRED)

add_executable( detect src/detect.cpp)
target_link_libraries( detect chilitags_static )
target_link_libraries( detect ${OpenCV_LIBS} )
target_link_libraries( detect pthread )
target_include_directories( detect PRIVATE /usr/local/include/chilitags/)

add_executable( tracking src/tracking.cpp)
target_link_libraries( tracking chilitags_static )
target_link_libraries( tracking ${OpenCV_LIBS} )
target_link_libraries( tracking pthread )
target_include_directories( tracking PRIVATE /usr/local/include/chilitags/)


add_executable( relative src/relative.cpp)
target_link_libraries( relative chilitags_static )
target_link_libraries( relative ${OpenCV_LIBS} )
target_link_libraries( relative pthread )
target_include_directories( relative PRIVATE /usr/local/include/chilitags/)
