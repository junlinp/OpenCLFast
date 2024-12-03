#include "opencv2/opencv.hpp"
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <iostream>
#include "opencl_helper.h"

cv::Ptr<cv::FastFeatureDetector> fastDetector = cv::FastFeatureDetector::create(80, true);

char PROGRAM_CONTEXT[] = "\n"
                         "__kernel void TwoSum(__global float* data_a, __global float* data_b,__global float* sum_result) {\n"
                         "              uint global_x = get_global_id(0);\n"
                         "              uint local_x = get_local_id(0);\n"
                         "              uint index = global_x;\n"
                         "              printf(\"opencl index %d\", index);\n"
                         "              sum_result[index] = data_a[index] + data_b[index];\n"
                         "}\n"
                         ;

char* KERNEL_FUNC = "TwoSum";

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " path/to/images";
    return 1;
  }
  cv::Mat img = cv::imread(argv[1]);
  cv::Mat image_gray;
  cv::cvtColor(img, image_gray, cv::COLOR_BGR2GRAY);

  cv::imwrite("gray.png", image_gray);

  std::vector<cv::KeyPoint> keypoints;
  fastDetector->detect(image_gray, keypoints);

  cv::Mat fast_img;
  cv::drawKeypoints(img, keypoints, fast_img);
  cv::imwrite("fast_opencv.png", fast_img);
  std::cout << "Detect : " << keypoints.size() << std::endl;


  OpenCL::OpenCLHelper opencl_helper;
  opencl_helper.BuildProgramFromSource(PROGRAM_CONTEXT, sizeof(PROGRAM_CONTEXT));

  const size_t ARRAY_SIZE = 384;
  float data_a[ARRAY_SIZE];
  float data_b[ARRAY_SIZE];
  float sum[ARRAY_SIZE];

  for (int i = 0; i < ARRAY_SIZE; i++) {
    data_a[i] = i;
    data_b[i] = i * 3;
  }

    // cl_mem input_a_buffer = opencl_helper.CreateBufferRead(sizeof(float) * ARRAY_SIZE);
    // cl_mem input_b_buffer = opencl_helper.CreateBufferRead(sizeof(float) * ARRAY_SIZE);
    // cl_mem sum_buffer = opencl_helper.CreateBufferReadWrite(sizeof(float) * ARRAY_SIZE);

    //cl_command_queue command_queue = clCreateCommandQueue(context,dev, 0, &err);

}

