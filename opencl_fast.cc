#include "opencv2/opencv.hpp"
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <iostream>
#include "opencl_helper.h"

#include "cpu_fast.h"


cv::Ptr<cv::FastFeatureDetector> fastDetector = cv::FastFeatureDetector::create(10, true);


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


// Detect FAST corners using CPU implementation
cv::Mat cpu_output;
DetectFASTCornersWithNMS(image_gray, cpu_output, 10);

// Convert detected corners to keypoints
std::vector<cv::KeyPoint> cpu_keypoints;
for(int row = 0; row < cpu_output.rows; row++) {
    for(int col = 0; col < cpu_output.cols; col++) {
        if(cpu_output.at<uchar>(row, col) > 0) {
            cpu_keypoints.push_back(cv::KeyPoint(col, row, 3));
        }
    }
}

// Draw keypoints
cv::Mat fast_cpu_img;
cv::drawKeypoints(img, cpu_keypoints, fast_cpu_img);
cv::imwrite("fast_cpu.png", fast_cpu_img);
std::cout << "CPU Detect : " << cpu_keypoints.size() << std::endl;

OpenCL::OpenCLFast(image_gray, "../fast.cl", "opencl_output.png");
}

