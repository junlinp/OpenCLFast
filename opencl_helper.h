
#define CL_TARGET_OPENCL_VERSION 110
#include <CL/cl.h>
#include <vector>
#include <string>
#include "iostream"

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

}