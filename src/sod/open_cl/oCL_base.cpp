#include "oCL_base.h"
#include "clError.h"
#include "../oCL_Kernels/oCL_Kernels.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <Rcpp.h>

const unsigned int default_local_item_size = 128;

OCL_base::OCL_base(const char* kernel_source, const char* kernel_name, bool compile_source)
{
  platform_id = NULL;
  device_id = NULL;
  num_devices = 0;
  num_platforms = 0;
  
  context = NULL;
  command_que = NULL;
  program = NULL;
  kernel = NULL;
  
  local_item_size = default_local_item_size;
  
  std::string define_statements("");  // empty string

  init_kernel(kernel_source, kernel_name, define_statements, compile_source);
}

OCL_base::OCL_base(const char* kernel_source, const char* kernel_name,
		   std::string define_statements, bool compile_source)
{
  platform_id = NULL;
  device_id = NULL;
  num_devices = 0;
  num_platforms = 0;
  
  context = NULL;
  command_que = NULL;
  program = NULL;
  kernel = NULL;
  
  local_item_size = default_local_item_size;

  init_kernel(kernel_source, kernel_name, define_statements, compile_source);
}

OCL_base::~OCL_base()
{
  // all of the following return a cl_uint error code
  // we should naturally check this, but for first implementation
  // let's try without.

  Rprintf("OCL_base destructor ? incurred here\n");

  clFlush(command_que);
  clFinish(command_que);
  
  cl_int ret;

  ret = clReleaseKernel(kernel);
  if(ret) report_error_pf("release kernel", ret);
  ret = clReleaseProgram(program);
  if(ret) report_error_pf("release program", ret);
  ret = clReleaseCommandQueue(command_que);
  if(ret) report_error_pf("release command_queue", ret);
  ret = clReleaseContext(context);
  if(ret) report_error_pf("releaseContext", ret);

}

void OCL_base::set_local_item_size(size_t li_size)
{
  local_item_size = li_size;
}

void OCL_base::device_properties()
{
  cl_int ret = 0;
  cl_uint compute_unit_no = 0;
  ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_unit_no, NULL);
  cl_ulong global_mem_size = 0;
  ret = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
  cl_ulong local_mem_size = 0;
  ret = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
  size_t device_max_work_group_size = 0;
  ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &device_max_work_group_size, NULL);
  cl_ulong device_max_constant_buffer_size = 0;
  ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &device_max_constant_buffer_size, NULL);

  Rprintf("Device properties:\nMax compute units\t%d\nMax work group\t%d\nGlobal memory\t%d\nMax_constant\t%d\nLocal memory\t%d\n",
	  compute_unit_no, device_max_work_group_size, global_mem_size, 
	  device_max_constant_buffer_size, local_mem_size);
    /*
  std::cout << "Device properties:" << std::endl
	    << "Max compute units: " << compute_unit_no << std::endl
	    << "Max work group   : " << device_max_work_group_size << std::endl
	    << "Global memory    : " << global_mem_size << std::endl
	    << "Max_constant     : " << device_max_constant_buffer_size << std::endl
    	    << "Local memory     : " << local_mem_size << std::endl;
    */
}

void OCL_base::kernel_properties()
{
  if(!device_id || !kernel){
    Rprintf("kernel_properties: no known kernel or device\n");
    return;
  }
  //size_t global_work_size[3];
  size_t work_group_size;
  size_t compile_work_group_size[3];
  cl_ulong local_mem_size;
  size_t preferred_work_group_size_multiple;
  cl_ulong private_mem_size;

  // don't bother checking
  //clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_GLOBAL_WORK_SIZE, sizeof(global_work_size), &global_work_size, NULL);
  clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(work_group_size), &work_group_size, NULL);
  clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, 
			   sizeof(compile_work_group_size), &compile_work_group_size, NULL);
  clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
  clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
			   sizeof(preferred_work_group_size_multiple), &preferred_work_group_size_multiple, NULL);
  clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(private_mem_size), &private_mem_size, NULL);

  /*
  std::cout << "Kernel properties:\n"
    //	    << "KERNEL_GLOBAL_WORK_SIZE            : " << global_work_size[0] << " : " 
    //	    << global_work_size[1] << " : " << global_work_size[2] << "\n"
	    << "KERNEL_WORK_GROUP_SIZE             : " << work_group_size << "\n"
	    << "KERNEL_COMPILE_WORK_GROUP_SIZE     : " << compile_work_group_size[0] << " : "
	    << compile_work_group_size[1] << " : " << compile_work_group_size[2] << "\n"
	    << "KERNEL_LOCAL_MEM_SIZE              : " << local_mem_size << "\n"
	    << "PREFERRED_WORK_GROUP_SIZE_MULTIPLE : " << preferred_work_group_size_multiple << "\n"
	    << "KERNEL_PRIVATE_MEM_SIZE            : " << private_mem_size << std::endl;
  */
  // Here I do rather wish that I could use the std::cout
  Rprintf("Kernel properties:\n");
  Rprintf("KERNEL_WORK_GROUP_SIZE\t%d\n", work_group_size);
  Rprintf("KERNEL_COMPILE_WORK_GROUP_SIZE\t%d\t%d\t%d\n", 
	  compile_work_group_size[0], compile_work_group_size[1], compile_work_group_size[2]);
  Rprintf("KERNEL_LOCAL_MEM_SIZE\t%d\n", local_mem_size);
  Rprintf("PREFERRED_WORK_GROUP_SIZE_MULTIPLE\t%d\n", preferred_work_group_size_multiple);
  Rprintf("KERNEL_PRIVATE_MEM_SIZE\t%d\n", private_mem_size);

}

char* make_source(std::string prefix, const char* const s_code, size_t& buf_length){
  buf_length = 1 + prefix.length() + 1 + strlen(s_code);
  char* buffer = new char[ buf_length ];
  memcpy((void*)buffer, (void*)prefix.c_str(), prefix.length());
  buffer[prefix.length()] = '\n';
  memcpy((void*)(buffer + prefix.length() + 1), (void*)s_code, strlen(s_code));
  buffer[buf_length-1] = 0;
  return(buffer);
}
  

void OCL_base::init_kernel(const char* kernel_source, const char* kernel_name, 
			   std::string define_statements, bool compile_source)
{
  if(!compile_source){
    Rprintf("Binary sources not supported yet\n");
    return;
  }  

  char* kernel_buffer = 0;
  size_t k_buffer_size=0;
  // if the kernel source has been predefined in "../oCL_Kernels/oCL_Kernels.h"
  // then use that.
  if(!strcmp(kernel_source, "move_deltoids")){  // strcmp returns 0 for equal
    kernel_buffer = make_source(define_statements, move_deltoids, k_buffer_size);
  }
  if(!strcmp(kernel_source, "move_deltoids_2")){
    kernel_buffer = make_source(define_statements, move_deltoids_2, k_buffer_size);
  }
  if(!strcmp(kernel_source, "move_deltoids_dummy")){
    kernel_buffer = make_source(define_statements, move_deltoids_dummy, k_buffer_size);
  }
    

  // if kernel buffer is not defined yet, try to read it from a file
  if(!kernel_buffer){
    std::ifstream in(kernel_source, std::ios::binary);
    if(!in){
      Rprintf("Unable to open kernel source file\n");
      return;
    }
    in.seekg(0, std::ios::end);
    ssize_t end_pos = in.tellg();
    Rprintf("in.tellg() reports : %d\n", end_pos);
    in.seekg(0, std::ios::beg);
    if(!end_pos){
      Rprintf("Unable to read from kernel source file\n");
      return;
    }
    // prepend #define statements to the kernal source.
    // add one extra \n to the 
    
    k_buffer_size = 1 + define_statements.size() + (size_t)end_pos + 1;
    kernel_buffer = new char[ k_buffer_size ];
    memset((void*)kernel_buffer, 0, sizeof(char) * k_buffer_size);
    //  kernel_buffer[k_buffer_size] = 0;  // this may not be needed.
    
    size_t copied_bytes = define_statements.copy(kernel_buffer, define_statements.size());
    if(copied_bytes != define_statements.size()){
      Rprintf("Unable to copy the full define statements\n");
      delete []kernel_buffer;
      return;
    }

    kernel_buffer[ define_statements.size() ] = '\n'; // add a new line for safety

    in.read((kernel_buffer + 1 + define_statements.size()), end_pos);
    if(in.gcount() != end_pos){
      Rprintf("Unable to read to end of kernel source file: %d != %d\n",
	      end_pos, in.gcount());
      delete []kernel_buffer;
      return;
    }
  }
  // at this point kernel_buffer should be defined.
  // and we can use it to compile the kernel.

  cl_int ret = 0; // return value. Use the same for all.
  
  ret = clGetPlatformIDs(1, &platform_id, &num_platforms);
  if(ret) report_error_pf("clGetPlatformIDs", ret);
  ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1,
			&device_id, &num_devices );
  if(ret) report_error_pf("clGetDeviceIDs", ret);
  if(ret != CL_SUCCESS){
    Rprintf("clGetDevice returned with error: %d\n", (int)ret);
    return;
  }

  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
  report_error_pf("clCreateContext", ret);

  command_que = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
  report_error_pf("clCreateCommandQueue", ret);

  program = clCreateProgramWithSource(context, 1, (const char**)&kernel_buffer, 
				      (const size_t*)&k_buffer_size, &ret);
  report_error_pf("clCreateProgramWithSource", ret);

  
  clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  report_error_pf("clBuildProgram", ret);

  char* build_log = NULL;
  size_t log_size = 1000;
  ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, build_log, &log_size);
  report_error_pf("clGetProgramBuildInfo", ret);
  build_log = new char[log_size+1];
  ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
  report_error_pf("clGetProgramBuildInfo", ret);
  
  // the below doesn't make much sense. But log_size is never 0, so, need to do something
  Rprintf("After building the kernel the log size is : %d\n", log_size);
  if(log_size > 2){
    Rprintf("clBuildProgram Error encountered:\n%s\n", build_log);
    Rprintf(".....\n%s\n....\n", kernel_buffer);
  }
  delete []build_log;

  kernel = clCreateKernel(program, kernel_name, &ret);
  report_error_pf("clCreateKernel", ret);
  
  delete []kernel_buffer;
  
}

cl_ulong OCL_base::time_command(cl_event* event)
{
  cl_ulong start = 0;
  cl_ulong end = 0;
  cl_int ret_value = 0;
  ret_value = clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
  if(ret_value) report_error_pf("Time event start", ret_value);
  ret_value = clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
  if(ret_value) report_error_pf("Time event end", ret_value);
  return(end-start);

}
