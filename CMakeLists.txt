cmake_minimum_required(VERSION 2.8)
project(Calibration)
find_package(OpenCV REQUIRED)

add_executable( calibrate src/calibrate.cpp)
target_link_libraries( calibrate chilitags_static )
target_link_libraries( calibrate ${OpenCV_LIBS} )
#target_include_directories( calibrate /usr/local/include/chilitags/)
