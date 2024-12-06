#include "opencv2/opencv.hpp"
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <iostream>
#include "opencl_helper.h"

cv::Ptr<cv::FastFeatureDetector> fastDetector = cv::FastFeatureDetector::create(10, true);

char PROGRAM_CONTEXT[] = "\n"
                         "__kernel void TwoSum(__global float* data_a, __global float* data_b,__global float* sum_result) {\n"
                         "              uint global_x = get_global_id(0);\n"
                         "              uint local_x = get_local_id(0);\n"
                         "              uint index = global_x;\n"
                         "              sum_result[index] = data_a[index] + data_b[index];\n"
                         "}\n"
                         ;

char* KERNEL_FUNC = "TwoSum";


bool FindCornor(cv::Mat img, size_t row, size_t col) {

  if (row < 3 || col < 3) {
    return false;
  }

  if (row > img.rows - 3) {
    return false;
  }

  if (col > img.cols - 3) {
    return false;
  }

  
}

void FastCPU(cv::Mat img, std::string output_file) {
  cv::Mat res(img.rows, img.cols, CV_8UC1);

  for (int row = 0; row < img.rows; row++) {
    for (int col = 0; col < img.cols; col++) {
      if (FindCornor(img, row, col)) {
        res.at<uchar>(row, col) = 255;
      } else {
        res.at<uchar>(row, col) = 0;
      }
    }
  }
  cv::imwrite(output_file, res);
}

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
  std::cout << "Image Width : " << img.cols << std::endl;
  std::cout << "Image Height : " << img.rows << std::endl;

  uint8_t* gray_image_data = new uint8_t[img.cols * img.rows];

  std::memcpy(gray_image_data, image_gray.data, img.cols * img.rows);

  size_t image_width = img.cols;
  size_t image_height = img.rows;


  OpenCL::OpenCLHelper opencl_helper;
  auto program = opencl_helper.BuildProgramFromSource(PROGRAM_CONTEXT,
                                                      sizeof(PROGRAM_CONTEXT));

  const size_t ARRAY_SIZE = 384;
  float data_a[ARRAY_SIZE];
  float data_b[ARRAY_SIZE];
  float sum[ARRAY_SIZE];

  for (int i = 0; i < ARRAY_SIZE; i++) {
    data_a[i] = i;
    data_b[i] = i * 3;
  }

  cl_mem input_a_buffer = opencl_helper.CreateBufferRead(sizeof(float) * ARRAY_SIZE);
  cl_mem input_b_buffer = opencl_helper.CreateBufferRead(sizeof(float) * ARRAY_SIZE);
  cl_mem sum_buffer = opencl_helper.CreateBufferReadWrite(sizeof(float) * ARRAY_SIZE);


  opencl_helper.CopyFromHost(input_a_buffer, data_a, sizeof(float) * ARRAY_SIZE);
  opencl_helper.CopyFromHost(input_b_buffer, data_b, sizeof(float) * ARRAY_SIZE);

  auto kernel = opencl_helper.CreateKernel(program, KERNEL_FUNC);

  opencl_helper.KernelBindArgs(kernel, input_a_buffer, input_b_buffer, sum_buffer);

  opencl_helper.KernelRun(kernel, ARRAY_SIZE, 1, 1);

  opencl_helper.CopyToHost(sum_buffer, sum, sizeof(float) * ARRAY_SIZE);

  for (int i = 0; i < ARRAY_SIZE; i++) {
    if (fabs(sum[i] - data_a[i] - data_b[i]) > 0.01 * fabs(data_a[i] + data_b[i])) {
      std::cout << "Error Result" << std::endl;
    }
  }
  std::cout << "Success" << std::endl;

  auto image_copy_program = opencl_helper.BuildProgramFromSourceFile(argv[2]);

  cl_mem image_buffer = opencl_helper.CreateOpenCLImage2D(
      image_width, image_height, OpenCL::ImageFormat::GrayUInt8,
      gray_image_data);

  cl_mem output_buffer = opencl_helper.CreateBufferReadWrite(image_width * image_height);

  auto image_copy_kernel = opencl_helper.CreateKernel(image_copy_program, "ImageCopy");

  opencl_helper.KernelBindArgs(image_copy_kernel, image_buffer, output_buffer, image_width, image_height);

  opencl_helper.KernelRun(image_copy_kernel, image_width, image_height, 1);

  char* output_image_buffer = new char[image_width * image_height];
  opencl_helper.CopyToHost(output_buffer, output_image_buffer, image_width * image_height);
  cv::Mat opencl_output(image_width, image_height, CV_8UC1,output_image_buffer);

  cv::imwrite("opencl_output.png", opencl_output);
  delete [] output_image_buffer;

}

