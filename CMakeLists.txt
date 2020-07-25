cmake_minimum_required(VERSION 2.8)

project( main )

find_package( OpenCV 4.1.2 REQUIRED )
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindPylon.cmake")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${Pylon_INCLUDE_DIRS} )

add_executable( main src/main.cpp )

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries( main 
    ${OpenCV_LIBS} 
    ${Pylon_LIBRARIES} 
    Threads::Threads)