#include "opencl_helper.h"
#include <string>
#include <iostream>
#include <vector>
#include <fstream>

namespace OpenCL {

OpenCLHelper::OpenCLHelper(OpenCLDeviceType type) {
    SelectPlatform();
    PlatformInfo(platforms_[0]);

    SelectDevice(platforms_[0], type);
    DeviceInfo(device_id_);
    CreateContextAndCommandQueue();
}


void OpenCLHelper::PlatformInfo(cl_platform_id platform_id) {
  char str_buffer[1024];
  int err = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME,
                          sizeof(str_buffer), &str_buffer, NULL);

  CheckError("PLATFORM_NAME", err);

  std::cout << "[Platform " << (platform_id)
            << "] CL_PLATFORM_NAME: " << str_buffer << std::endl;

  err = clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, sizeof(str_buffer),
                          &str_buffer, NULL);

  CheckError("PLATFORM_VENDOR", err);

  std::cout << "[Platform " << platform_id
            << "] CL_PLATFORM_VENDOR: " << str_buffer << std::endl;

  err = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION,
                          sizeof(str_buffer), &str_buffer, NULL);

  std::cout << "[Platform " << platform_id
            << "] CL_PLATFORM_VERSION: " << str_buffer << std::endl;

  err = clGetPlatformInfo(platform_id, CL_PLATFORM_PROFILE,
                          sizeof(str_buffer), &str_buffer, NULL);

  std::cout << "[Platform " << platform_id <<
           "] CL_PLATFORM_PROFILE: " << str_buffer << std::endl;

  err = clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS,
                          sizeof(str_buffer), &str_buffer, NULL);

  std::cout << "[Platform " << platform_id
            << "] CL_PLATFORM_EXTENSIONS: " << str_buffer << std::endl;


}

void OpenCLHelper::SelectDevice(cl_platform_id platform_id,
                                OpenCLDeviceType device_type) {
  cl_uint num_devices_available;
  int err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices_available);
  CheckError("clGetDeviceIDs", err);

  if (num_devices_available < 1) {
    std::cerr << "No device";
    exit(1);
  }
  cl_device_id cl_devices[num_devices_available];
  switch (device_type) {
    case OpenCL::OpenCLDeviceType::CPU: {
      int err = clGetDeviceIDs(platform_id,CL_DEVICE_TYPE_CPU, 0, NULL,
                               &num_devices_available);
      CheckError("clGetDeviceIDs", err);
      err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU,
                           num_devices_available, cl_devices, NULL);
      CheckError("clGetDeviceIDs", err);
    }

    case OpenCL::OpenCLDeviceType::GPU: {
      int err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL,
                               &num_devices_available);
      CheckError("clGetDeviceIDs", err);
      err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU,
                           num_devices_available, cl_devices, NULL);
      CheckError("clGetDeviceIDs", err);
    }
  }
  device_id_ = cl_devices[0];
}

void OpenCLHelper::DeviceInfo(cl_device_id device_id) {

  char str_buffer[1024];
  int err = clGetDeviceInfo(device_id, CL_DEVICE_NAME,
                        sizeof(str_buffer), &str_buffer, NULL);

  CheckError("clGetDeviceInfo", err);

  std::cout << " CL_DEVICE_NAME: " << str_buffer << std::endl;

  err = clGetDeviceInfo(device_id, CL_DEVICE_VERSION, sizeof(str_buffer),
                        &str_buffer, NULL);

  CheckError("clGetDeviceInfo", err);
  std::cout << " CL_DEVICE_VERSION: " << str_buffer << std::endl;

  err = clGetDeviceInfo(device_id, CL_DRIVER_VERSION, sizeof(str_buffer),
                        &str_buffer, NULL);

  std::cout << " CL_DRIVER_VERSION: " << str_buffer << std::endl;

  err = clGetDeviceInfo(device_id, CL_DEVICE_OPENCL_C_VERSION,
                        sizeof(str_buffer), &str_buffer, NULL);

  std::cout << " CL_DEVICE_OPENCL_C_VERSION: " << str_buffer << std::endl;

  cl_uint max_compute_units_availabel;
  err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(max_compute_units_availabel),
                        &max_compute_units_availabel, NULL);

  std::cout << " CL_DEVICE_MAX_COMPUTE_UNITS: " +
                   std::to_string(max_compute_units_availabel) + "\n";

  cl_ulong global_mem_size;
  err = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE,
                        sizeof(global_mem_size), &global_mem_size, NULL);

  std::cout << " CL_DEVICE_GLOBAL_MEM_SIZE: " + std::to_string(global_mem_size)
            << std::endl;

  size_t max_work_group_size;
  err =
      clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                      sizeof(max_work_group_size), &max_work_group_size, NULL);
      std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " + std::to_string(max_work_group_size) << std::endl;

  cl_uint max_work_item_dims;
  err = clGetDeviceInfo(device_id,
                        CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                        sizeof(max_work_item_dims), &max_work_item_dims, NULL);

  std::cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: "<< max_work_item_dims << std::endl;

  size_t work_item_sizes[max_work_item_dims];

  err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                        sizeof(work_item_sizes), &work_item_sizes, NULL);

  std::cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << std::endl;
  for (size_t work_item_dim = 0; work_item_dim < max_work_item_dims;
       work_item_dim++) {
    std::cout << std::to_string(work_item_sizes[work_item_dim]) << std::endl;
  }
}


void OpenCLHelper::SelectPlatform() {
    cl_uint num_platforms_available;
    int err;
    // query available platform
    err = clGetPlatformIDs(0, NULL, &num_platforms_available);
    CheckError("clGetPlatformIDsAvailable", err);


    platforms_.resize(num_platforms_available);
    err = clGetPlatformIDs(num_platforms_available, platforms_.data(), NULL);
    CheckError("clGetPlatformIDs", err);

}

cl_program
OpenCLHelper::BuildProgramFromSource(const char *program_content,
                                     size_t progmran_content_length) {
  return BuildProgramFromSourceInternal(ctx_, device_id_, program_content,
                                        progmran_content_length);
}

cl_program OpenCLHelper::BuildProgramFromSourceInternal(
    cl_context ctx, cl_device_id device_id, const char *program_content,
    size_t progmran_content_length) {
  int err;
  cl_program program = clCreateProgramWithSource(
      ctx, 1, (const char **)&program_content,
      static_cast<const size_t *>(&progmran_content_length), &err);
  CheckError("CreateProgramWithSource", err);
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    char *program_log = (char *)malloc(log_size + 1);
    program_log[log_size] = 0;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                          log_size + 1, program_log, NULL);
    std::cerr << "OpenCLBuildProgram" << "OpenCL Build Program Error : %s\n"
              << program_log;
    free(program_log);
    exit(1);
  }
  return program;
}

cl_program OpenCLHelper::BuildProgramFromSourceFile(const std::string& file_path) {
  std::ifstream ifs(file_path);
  if (!ifs) {
    std::cerr << "Source file : " << file_path << " can't open.";
    exit(1);
  }

  ifs.seekg(0, std::ios::end);
  size_t length = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  char* buf = new char[length];
  ifs.read(buf, length);

  auto program = BuildProgramFromSource(buf, length);

  delete[] buf;
  return program;
}


void OpenCLHelper::CreateContextAndCommandQueue() {
  int err;
  ctx_ = clCreateContext(NULL, 1, &device_id_, NULL, NULL, &err);
  CheckError("clCreateContext", err);

  command_queue_ = clCreateCommandQueue(ctx_, device_id_, 0, &err);

  CheckError("clCreateCommandQueue", err);
}

cl_mem OpenCLHelper::CreateBufferRead(size_t memory_size_bytes) {
  cl_int err;
  cl_mem ret_mem =
      clCreateBuffer(ctx_, CL_MEM_READ_ONLY, memory_size_bytes, nullptr, &err);
  CheckError("CreateMemoryRead", err);
  return ret_mem;
}
cl_mem OpenCLHelper::CreateBufferReadWrite(size_t memory_size_bytes) {

  cl_int err;
  cl_mem ret_mem =
      clCreateBuffer(ctx_, CL_MEM_READ_WRITE, memory_size_bytes, nullptr, &err);
  CheckError("CreateMemoryREADWRITE", err);
  return ret_mem;
}

cl_mem OpenCLHelper::CreateOpenCLImage2D(size_t width, size_t height, ImageFormat image_format, void* host_ptr) {
  cl_int error = CL_SUCCESS;

  cl_image_format opencl_image_format;
  switch (image_format) {
  case OpenCL::ImageFormat::GrayUInt8: {

    opencl_image_format.image_channel_order = CL_R;
    opencl_image_format.image_channel_data_type = CL_UNSIGNED_INT8;
    break;
  }

  }

  cl_mem ret_mem = clCreateImage2D(ctx_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &opencl_image_format,
                                   width, height, 0, host_ptr, &error);
  CheckError("CreateImage2D", error);
  return ret_mem;
}

void OpenCLHelper::CopyFromHost(cl_mem device_memory, void *host_ptr,
                                size_t host_ptr_length) {
  int error = clEnqueueWriteBuffer(command_queue_, device_memory, CL_TRUE, 0,
                                   host_ptr_length, host_ptr, 0, NULL, NULL);
}

void OpenCLHelper::CopyToHost(cl_mem device_memory, void *host_ptr,
                              size_t host_ptr_length) {

  int error = clEnqueueReadBuffer(command_queue_, device_memory, CL_TRUE, 0,
                                  host_ptr_length, host_ptr, 0, NULL, NULL);
}

cl_kernel OpenCLHelper::CreateKernel(cl_program program, const std::string& kernel_function_name) {
  int error;

  cl_kernel kernel = clCreateKernel(program, kernel_function_name.c_str(), &error);
  CheckError("CreateKernel", error);
  return kernel;
}


void OpenCLHelper::KernelRun(cl_kernel kernel, size_t global_group_x, size_t global_group_y, size_t global_group_z) {
    size_t global_sizes[3] = {global_group_x, global_group_y, global_group_z};
    //size_t local_size = 8;
    //cl_event kernel_event;
    // Enqueue kernel
    int err = clEnqueueNDRangeKernel(command_queue_, kernel, 3, NULL,
                                     global_sizes, NULL, 0, NULL, NULL);
    CheckError("EnqueueNDRangeKernel", err);
}

}