cmake_minimum_required(VERSION 3.13)
project(OpenCLFast)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    add_executable(fast opencl_fast.cc opencl_helper.cc cpu_fast.h)

include_directories(
/usr/local/include/opencv4
/usr/local/cuda-12.5/include
)


find_package(OpenCV REQUIRED)
if(APPLE)
    # Try to find OpenCL package first
    find_package(OpenCL)
    
    if(NOT OpenCL_FOUND)
        # If not found through find_package, try common Mac locations
        find_library(OpenCL_LIBRARIES
            NAMES OpenCL
            PATHS
            "/System/Library/Frameworks/OpenCL.framework"
            "/Library/Frameworks/OpenCL.framework"
        )
        
        if(OpenCL_LIBRARIES)
            set(OpenCL_FOUND TRUE)
            get_filename_component(OpenCL_FRAMEWORK_DIR "${OpenCL_LIBRARIES}" DIRECTORY)
            set(OpenCL_INCLUDE_DIRS "${OpenCL_FRAMEWORK_DIR}/Headers")
        endif()
    endif()

    if(NOT OpenCL_FOUND)
        message(FATAL_ERROR "OpenCL not found!")
    endif()

    include_directories(${OpenCL_INCLUDE_DIRS})
    target_link_libraries(fast ${OpenCL_LIBRARIES})
endif()
if(UNIX AND NOT APPLE)
    find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIRS})
    link_directories(/usr/local/cuda/lib64)
    target_link_libraries(fast ${OpenCL_LIBRARIES})
endif()

link_directories(
/usr/local/lib
/usr/lib
/usr/lib/x86_64-linux-gnu
${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
    fast
    ${OpenCV_LIBS}
    )



