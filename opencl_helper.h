#define CL_TARGET_OPENCL_VERSION 110
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#elif defined(__linux__)
#include <CL/cl.h>
#else
#error "Unsupported platform"
#endif

#include <vector>
#include <string>
#include "iostream"
#include <opencv2/opencv.hpp>
namespace {

void CheckError(std::string tag, int error_code) {
  if (error_code != CL_SUCCESS) {
    std::cerr << tag << " error " << error_code << std::endl;
    exit(error_code);
  }
}

}
namespace OpenCL {

enum OpenCLDeviceType {
    CPU = 0,
    GPU = 1,
};

enum ImageFormat {
  // 
  GrayUInt8 = 0,
};

class OpenCLHelper {
public:
    explicit OpenCLHelper(OpenCLDeviceType = OpenCLDeviceType::GPU);

    cl_program BuildProgramFromSource(const char* program_content, size_t progmran_content_length);
    cl_program BuildProgramFromSourceFile(const std::string& file_path);

    cl_mem CreateBufferRead(size_t memory_size_bytes);
    cl_mem CreateBufferReadWrite(size_t memory_size_bytes);
    // host_ptr should contains width * height * sizeof(ImageFormat data size)
    cl_mem CreateOpenCLImage2D(size_t width, size_t height, ImageFormat image_format, void* host_ptr);

    void CopyFromHost(cl_mem device_memory, void* host_ptr, size_t host_ptr_length);
    void CopyToHost(cl_mem device_memory, void* host_ptr, size_t host_ptr_length);

    cl_kernel CreateKernel(cl_program program, const std::string& kernel_function_name);

    template<class... ARGS>
    void KernelBindArgs(cl_kernel kernel, ARGS... args) {
      int arg_index = 0;
      (CheckError("KernelSetArg_" + std::to_string(arg_index),
                  clSetKernelArg(kernel, arg_index++, sizeof(args), &args)),
       ...);
    }

    void KernelRun(cl_kernel kernel, size_t global_group_x, size_t global_group_y, size_t global_group_z);

private:
    void SelectPlatform();
    void PlatformInfo(cl_platform_id platform_id);

    void SelectDevice(cl_platform_id platform_id, OpenCLDeviceType device_type);

    void DeviceInfo(cl_device_id device_id);

    void CreateContextAndCommandQueue();


    cl_program BuildProgramFromSourceInternal(cl_context ctx, cl_device_id device_id, const char* program_content, size_t progmran_content_length);

    std::vector<cl_platform_id> platforms_;
    cl_device_id device_id_;
    cl_context ctx_;
    cl_command_queue command_queue_;
};

namespace {

char PROGRAM_CONTEXT[] = "\n"
                         "__kernel void TwoSum(__global float* data_a, __global float* data_b,__global float* sum_result) {\n"
                         "              uint global_x = get_global_id(0);\n"
                         "              uint local_x = get_local_id(0);\n"
                         "              uint index = global_x;\n"
                         "              sum_result[index] = data_a[index] + data_b[index];\n"
                         "}\n"
                         ;

char* KERNEL_FUNC = "TwoSum";
}

inline void OpenCLFast(cv::Mat img, std::string program_source_file, std::string output_file) {
  size_t image_width = img.cols;
  size_t image_height = img.rows;
  char* gray_image_data = new char[image_width * image_height];
  std::memcpy(gray_image_data, img.data, image_width * image_height);

  OpenCL::OpenCLHelper opencl_helper;
  auto program = opencl_helper.BuildProgramFromSourceFile(program_source_file);

  auto total_start_time = std::chrono::high_resolution_clock::now();

  // Create input image and output buffers
  auto mem_h2d_start = std::chrono::high_resolution_clock::now();
  
  cl_mem image_buffer = opencl_helper.CreateOpenCLImage2D(
      image_width, image_height, OpenCL::ImageFormat::GrayUInt8,
      gray_image_data);

  cl_mem corner_buffer = opencl_helper.CreateBufferReadWrite(image_width * image_height);
  cl_mem nms_buffer = opencl_helper.CreateBufferReadWrite(image_width * image_height);
  
  auto mem_h2d_end = std::chrono::high_resolution_clock::now();
  auto mem_h2d_duration = std::chrono::duration_cast<std::chrono::milliseconds>(mem_h2d_end - mem_h2d_start);

  // Run FAST corner detection
  auto fast_start_time = std::chrono::high_resolution_clock::now();
  
  auto fast_kernel = opencl_helper.CreateKernel(program, "FASTCorner");
  opencl_helper.KernelBindArgs(fast_kernel, image_buffer, corner_buffer, 10);
  opencl_helper.KernelRun(fast_kernel, image_width, image_height, 1);
  
  auto fast_end_time = std::chrono::high_resolution_clock::now();
  auto fast_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fast_end_time - fast_start_time);

  // Run non-maximum suppression
  auto nms_start_time = std::chrono::high_resolution_clock::now();
  
  auto nms_kernel = opencl_helper.CreateKernel(program, "NonMaximumSuppression");
  opencl_helper.KernelBindArgs(nms_kernel, corner_buffer, nms_buffer, 3);
  opencl_helper.KernelRun(nms_kernel, image_width, image_height, 1);
  
  auto nms_end_time = std::chrono::high_resolution_clock::now();
  auto nms_duration = std::chrono::duration_cast<std::chrono::milliseconds>(nms_end_time - nms_start_time);

  // Get results
  auto mem_d2h_start = std::chrono::high_resolution_clock::now();
  
  char* output_image_buffer = new char[image_width * image_height];
  opencl_helper.CopyToHost(nms_buffer, output_image_buffer,
                           image_width * image_height);
                           
  auto mem_d2h_end = std::chrono::high_resolution_clock::now();
  auto mem_d2h_duration = std::chrono::duration_cast<std::chrono::milliseconds>(mem_d2h_end - mem_d2h_start);

  auto total_end_time = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);
  
  cv::Mat opencl_output(image_height, image_width, CV_8UC1, output_image_buffer);
  cv::imwrite(output_file, opencl_output);

  // Convert detected corners to keypoints
  std::vector<cv::KeyPoint> opencl_keypoints;
  for(int row = 0; row < opencl_output.rows; row++) {
    for(int col = 0; col < opencl_output.cols; col++) {
      if(opencl_output.at<uchar>(row, col) > 0) {
        opencl_keypoints.push_back(cv::KeyPoint(col, row, 3));
      }
    }
  }

  // Draw keypoints
  cv::Mat fast_opencl_img;
  cv::drawKeypoints(img, opencl_keypoints, fast_opencl_img);
  cv::imwrite("fast_opencl.png", fast_opencl_img);

  std::cout << "OpenCL Detect : " << opencl_keypoints.size() << std::endl;
  std::cout << "OpenCL Memory H2D Runtime: " << mem_h2d_duration.count() << " ms" << std::endl;
  std::cout << "OpenCL Memory D2H Runtime: " << mem_d2h_duration.count() << " ms" << std::endl;
  std::cout << "OpenCL FAST Kernel Runtime: " << fast_duration.count() << " ms" << std::endl;
  std::cout << "OpenCL NMS Kernel Runtime: " << nms_duration.count() << " ms" << std::endl;
  std::cout << "OpenCL Total Runtime: " << total_duration.count() << " ms" << std::endl;

  delete [] gray_image_data;
  delete [] output_image_buffer;
}

}