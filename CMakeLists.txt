
    cmake_minimum_required(VERSION 3.13)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    add_executable(fast opencl_fast.cc opencl_helper.cc)

include_directories(
/usr/local/include/opencv4
/usr/local/cuda-12.5/include
)


find_package(OpenCV REQUIRED)

link_directories(
/usr/local/lib
/usr/lib
/usr/lib/x86_64-linux-gnu
${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
    fast
    /usr/local/cuda-12.5/lib64/libOpenCL.so
    ${OpenCV_LIBS}
    )
    
