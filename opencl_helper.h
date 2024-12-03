
#define CL_TARGET_OPENCL_VERSION 110
#include <CL/cl.h>
#include <vector>

namespace OpenCL {

enum OpenCLDeviceType {
    CPU = 0,
    GPU = 1,
};

class OpenCLHelper {
public:
    explicit OpenCLHelper(OpenCLDeviceType = OpenCLDeviceType::GPU);

    cl_program BuildProgramFromSource(const char* program_content, size_t progmran_content_length);

    cl_mem CreateBufferRead(size_t memory_size_bytes);
    cl_mem CreateBufferReadWrite(size_t memory_size_bytes);

private:
    void SelectPlatform();
    void PlatformInfo(cl_platform_id platform_id);

    void SelectDevice(cl_platform_id platform_id, OpenCLDeviceType device_type);

    void DeviceInfo(cl_device_id device_id);

    void CreateContext();

    cl_program BuildProgramFromSourceInternal(cl_context ctx, cl_device_id device_id, const char* program_content, size_t progmran_content_length);

    std::vector<cl_platform_id> platforms_;
    cl_device_id device_id_;
    cl_context ctx_;

};

}