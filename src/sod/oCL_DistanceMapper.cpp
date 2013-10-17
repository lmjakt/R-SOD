#include "oCL_DistanceMapper.h"
#include "open_cl/oCL_base.h"
#include "open_cl/clError.h"
#include <string.h>
#include <fstream>
#include <iostream>
#include <Rcpp.h>  // included for Rprintf and other standard methods.

// need to change these hardcoded paths into softcoded ones ? ?

//const char* kernel_source = "/home/martin/applications/R_packages/SOD/src/sod/oCL_Kernels/move_deltoids_dummy.cl";
//const char* kernel_source = "/home/martin/applications/R_packages/SOD/src/sod/oCL_Kernels/move_deltoids.cl";

//const char* const kernel_source = "/home/martin/applications/R_packages/SOD/src/sod/oCL_Kernels/move_deltoids_2.cl";
const char* const kernel_source = "move_deltoids_2"; 
const char* const kernel_name = "move_deltoids";

OCL_DistanceMapper::OCL_DistanceMapper(std::string def_statement)
  : OCL_base(kernel_source, kernel_name, def_statement, true)
{
  dimFactors = 0;
}

OCL_DistanceMapper::~OCL_DistanceMapper()
{
  if(dimFactors)
    delete []dimFactors;
}

MappingInfo OCL_DistanceMapper::reduce_dimensions(float* points, unsigned int node_no,
							     unsigned int starting_dimensionality,
							     unsigned int target_dimensionality,
							     unsigned int iterations,
							     float* distances)
{

  MappingInfo dummyInfo;
  std::vector<stressInfo> stress_data(iterations);
  // check whether kernel and stuff is set up correctly //
  if(!kernel)
    Rprintf("openCL: kernel not defined\n");
  if(!context)
    Rprintf("openCL: no context specified\n");
  if(!command_que)
    Rprintf("openCL: no command queue specified\n");
  if(!kernel || !context || !command_que)
    return(dummyInfo);
  
  if(!starting_dimensionality || !target_dimensionality){
    Rprintf("OCL_DistanceMapper null starting (%d) or target dimensionality (%d)\n", 
	    starting_dimensionality, target_dimensionality);
    return(dummyInfo);
  }

  if(target_dimensionality > starting_dimensionality){
    uint d = target_dimensionality;
    target_dimensionality = starting_dimensionality;
    starting_dimensionality = d;
  }
  if(!node_no){
    Rprintf("OCL_DistanceMapper: node_no is 0\n");
    return(dummyInfo);
  }
  // Set up the dimFactor vector defining the extent of different dimensions
  dimensionality = starting_dimensionality;
  t_dimensionality = target_dimensionality;
  if(dimFactors) delete []dimFactors;
  dimFactors = new float[dimensionality];
  for(uint i=0; i < dimensionality; ++i)
    dimFactors[i] = 1.0;
  
  // let's set up a global item size, as a multiple of local_item_size
  // at the moment local_item_size is hardcoded into the constructor, but
  // it can be changed via a call to setLocalItemSize()

  size_t global_item_size = local_item_size;
  while(global_item_size < node_no)
    global_item_size += local_item_size;
  
  std::cout << "local item size is : " << local_item_size << "\n"
	    << "global item size is: " << global_item_size << std::endl;

  // if we want to make use of reasonable calls to async_work_group_copy
  // we need to make sure that the data itself aligns to multiples of 
  // the local_item_size
  float* points_i = new float[ global_item_size * dimensionality ];
  memset((void*)points_i, 0,  sizeof(float) * global_item_size * dimensionality);
  memcpy((void*)points_i, (void*)points, sizeof(float) * node_no * dimensionality);

  // set up the appropriate buffers
  float* points_j = new float[ global_item_size * dimensionality ];
  memset((void*)points_j, 0, sizeof(float) * node_no * dimensionality);

  float* stress = new float[ node_no ];
  
  cl_int ret = 0;

  unsigned int* error_buffer = new unsigned int[node_no * node_no];

  // And then we need to create a set of mem objects
  // note we use NULL here since we are not asking the GPU to use
  // host memory
  cl_mem pos_i_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
				    sizeof(float) * global_item_size * dimensionality, NULL, &ret);
  cl_mem pos_j_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
				    sizeof(float) * global_item_size * dimensionality, NULL, &ret);
  cl_mem stress_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
				     sizeof(float) * node_no, NULL, &ret);
  cl_mem dim_factors_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
					  sizeof(float) * dimensionality, NULL, &ret);
  cl_mem distances_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
					sizeof(float) * node_no * node_no, NULL, &ret);
  
  cl_mem error_buffer_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
  					   sizeof(unsigned int) * node_no * node_no, NULL, &ret);
  
  if(ret) report_error_pf("end of clCreateBuffer section", ret);
  ret = 0;

  // some timing variables
  cl_ulong pre_mem_enque_time = 0;
  cl_ulong loop_mem_que_time = 0;
  cl_ulong mem_read_time = 0;
  cl_ulong kernel_time = 0;

  cl_event write_mem_event;
  //// enque the resulting buffers
  ret = clEnqueueWriteBuffer(command_que, pos_j_obj, CL_TRUE, 0,
			     sizeof(float) * global_item_size * dimensionality, (void*)points_j,
			     0, NULL, &write_mem_event);
  pre_mem_enque_time += time_command(&write_mem_event);

  ret = clEnqueueWriteBuffer(command_que, stress_obj, CL_TRUE, 0,
			     sizeof(float) * node_no, (void*)stress,
			     0, NULL, &write_mem_event);
  pre_mem_enque_time += time_command(&write_mem_event);

  ret = clEnqueueWriteBuffer(command_que, distances_obj, CL_TRUE, 0,
			     sizeof(float) * node_no * node_no, (void*) distances,
			     0, NULL, &write_mem_event);
  pre_mem_enque_time += time_command(&write_mem_event);
  
  ret = clEnqueueWriteBuffer(command_que, error_buffer_obj, CL_TRUE, 0,
  			     sizeof(unsigned int) * node_no * node_no, (void*)error_buffer,
  			     0, NULL, &write_mem_event);
  pre_mem_enque_time += time_command(&write_mem_event);
  
  // though this only checks the last command.
  // maybe better to make a wrapper function that checks.
  if(ret) report_error_pf("end of clEnqueueWriteBuffer section", ret);
  
  /// and then we can finally set the arguments, before looping through and getting
  /// the kernel to do it's thing.

  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&pos_j_obj);
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&stress_obj);
  ret = clSetKernelArg(kernel, 3, sizeof(uint), &dimensionality);
  // argument 4 set in the loop. 
  ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&distances_obj);
  ret = clSetKernelArg(kernel, 6, sizeof(uint), (void*)&node_no);
  ret = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&error_buffer_obj);

  if(ret) report_error_pf("end of clSetKernelArg section", ret);

  // initially write the loop in the stupid way.
  // 1. enque the initial points (pos_i_obj) using the points_i
  // 2. call clEnqueueNDRangeKernel
  // 3. Wait
  // 4. call clEnqueueReadBuffer to read pos_j_obj to points_i
  // 5. repeat from 1
  //
  // strictly we could just leave the memory on the GPU
  // but ask the kernel to map from i to j or j to i depending
  // on whether odd or even.
  // But I'm not sure how to make sure things are synchronised in this case.
  
  Rprintf("Starting the squeeze\n");
  // Let's print out 80 colums to indicate progress
  unsigned int columns = 80;
  unsigned int div = (iterations / columns);
  for(unsigned int i=0; i < iterations; ++i){
    if(!(i % div)) Rprintf("-");
  }
  Rprintf("|\n");

  for(unsigned int i=0; i < iterations; ++i){
    // progress bar
    if(!(i % div))
      Rprintf("=");
    // And the real stuff.. 
    ret = clEnqueueWriteBuffer(command_que, pos_i_obj, CL_TRUE, 0,
			       sizeof(float) * global_item_size * dimensionality, (void*)points_i,
			       0, NULL, &write_mem_event);
    loop_mem_que_time += time_command(&write_mem_event);
    
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&pos_i_obj);
    
    ret = clEnqueueWriteBuffer(command_que, dim_factors_obj, CL_TRUE, 0,
			       sizeof(float) * dimensionality, (void*)dimFactors,
			       0, NULL, &write_mem_event);
    loop_mem_que_time += time_command(&write_mem_event);
    
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&dim_factors_obj);
    
    // Run the kernel one iteration and measure the time taken
    cl_event move_event;
    ret = clEnqueueNDRangeKernel(command_que, kernel, 1, NULL,
				 &global_item_size, &local_item_size, 0, NULL, &move_event);
    if(ret) report_error_pf("clEnqueueNDRangeKernel", ret);
    
    ret = clWaitForEvents(1, &move_event);
    if(ret) report_error_pf("clWaitForEvents", ret);
    kernel_time += time_command(&move_event);
    
    // Then extract the relevant data from the kernel and swap buffers around
    
    // set the old points (points_i) to the new ones (points_j)
    cl_event read_event;
    ret = clEnqueueReadBuffer(command_que, pos_j_obj, CL_TRUE, 0,
			      sizeof(float) * global_item_size * dimensionality,
			      points_i, 0, NULL, &read_event);
    mem_read_time += time_command(&read_event);
    
    // read the stress vector
    ret = clEnqueueReadBuffer(command_que, stress_obj, CL_TRUE, 0,
			      sizeof(float) * node_no,
			      stress, 0, NULL, &read_event);
    mem_read_time += time_command(&read_event);
    
    float stress_sum = 0;
    for(uint j=0; j < node_no; ++j){
      stress_sum += stress[j];  // THE REAL CODE
    }
    stress_data[i].setStress(dimensionality, dimFactors, dimensionality, stress_sum); 
    
    shrink_dimensionality(iterations);  // this changes the dimFactors vector
  }
  Rprintf("\nSqueezing done\n");
    
  // let's write a table of positions that we can plot with R or something
  // Keep the below for troubleshooting purposes.  
  std::ofstream out("oCL_DM_positions.txt");
  if(!out){
    std::cerr << "oCL_DistanceMapper unable to open a file to stick numbers in" << std::endl;
  }else{
    for(uint i=0; i < node_no; ++i){
      out << i;
      for(uint j=0; j < dimensionality; ++j)
	out << "\t" << points_i[i * dimensionality + j];
      out << "\n";
    }
    out.close();
  }
  
  // Let's convert the stress values to a vector of floats
  std::vector<float> node_stress(node_no);
  node_stress.assign( stress, stress+node_no );
  
  delete []error_buffer; // and maybe the others as well?
  delete []points_j;
  //delete []points_i;
  delete []stress;
  
  // Encapsulate the timing data in a map
  std::map<std::string, float> time_data;
  time_data["Pre_mem_enque_time"] = (float)pre_mem_enque_time / 1e9;
  time_data["Loop_mem_enque_time"] = (float)loop_mem_que_time / 1e9;
  time_data["Kernel_time"] = (float)kernel_time / 1e9;
  time_data["Mem_read_time"] = (float)mem_read_time / 1e9;

  Rprintf("Time data (seconds)\n");
  Rprintf("Pre mem enque time  %f\n", (float)pre_mem_enque_time / 1e9);
  Rprintf("Loop mem enque time %f\n", (float)loop_mem_que_time / 1e9);
  Rprintf("Kernel time         %f\n", (float)kernel_time / 1e9);
  Rprintf("Mem read time       %f\n", (float)mem_read_time / 1e9);

  //std::cout << "Timing data in s" << std::endl;
  //std::cout << "Pre mem enque time : " << pre_mem_enque_time / 1e9 << "\n"
  //          << "Loop mem enque time: " << loop_mem_que_time / 1e9 << "\n"
  //	      << "Kernel time        : " << kernel_time / 1e9 << "\n"
  //          << "Mem read time      : " << mem_read_time / 1e9 << "\n";

  return( MappingInfo(node_no, dimensionality, points_i, stress_data, time_data, node_stress) );
}


// use a parallel shrink for dims above t_dimensionality
void OCL_DistanceMapper::shrink_dimensionality(unsigned int iter_no)
{
  float r_factor = (1.0 / 0.9) / (float)iter_no;
  for(uint i=t_dimensionality; i < dimensionality; ++i)
    dimFactors[i] = (dimFactors[i] - r_factor) < 0 ? 0 : (dimFactors[i] - r_factor);
  
}
