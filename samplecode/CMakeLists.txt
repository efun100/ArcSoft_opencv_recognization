cmake_minimum_required(VERSION 2.8)
project(FaceRecognize)

find_package(OpenCV REQUIRED)

include_directories(. ../inc ${OpenCV_INCLUDE_DIRS})
link_directories(../../lib/linux_x64)

add_executable(FaceRecognize
               arcsoft_afd_samplecode.cpp)
target_link_libraries(FaceRecognize ${OpenCV_LIBS})
target_link_libraries(FaceRecognize -larcsoft_fsdk_face_detection)
