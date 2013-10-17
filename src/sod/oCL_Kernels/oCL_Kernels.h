#ifndef OCL_KERNELS_H
#define OCL_KERNELS_H

const char* const move_deltoids = "\
// to make use of 'just in time' compiling by modifying the string. Quite easy to do, by simply\n\
// prepending a set of define statements.\n\
\n\
//This source needs to be prepended with\n\
// #define DIM_NO \n\
// #define NODE_NO\n\
\n\
__kernel void move_deltoids(__global float* pos_i, __global float* pos_j, __global float* stress, \n\
			    uint dim_no, __global float* dim_factors, __global float* distances, uint node_no,\n\
			    __global float* force_vector, __global unsigned int* error_buffer){\n\
\n\
  size_t g_i = get_global_id(0);\n\
  size_t l_i = get_local_id(0);\n\
\n\
  if(g_i >= node_no)\n\
    return;\n\
\n\
  float p_dim_factors[DIM_NO];\n\
  float p_fv[DIM_NO];\n\
  float coord_dist[DIM_NO];\n\
  float this_pos[DIM_NO];\n\
\n\
  __global float* p_j = pos_j + (g_i * dim_no);\n\
  float p_stress = 0;\n\
  \n\
  // first set the local force_vector to 0\n\
  // can't call any functions so need to iterate through.\n\
  for(uint i=0; i < dim_no; ++i){\n\
    p_fv[i] = 0;\n\
    p_dim_factors[i] = dim_factors[i];\n\
    this_pos[i] = *(pos_i + (g_i * dim_no + i));\n\
  }\n\
  // then go through each node; calculate the current euclidean distance and increment\n\
  // the force_vector as appropriate.\n\
  for(uint i=1; i < node_no; ++i){\n\
    // the problem here is that every gpu will want to access the same global variables here\n\
    // compare to 0, and then 1st position. This is likely to be bad. So, rather than that, do\n\
    //int ni = i;\n\
    int ni = (g_i + i) % node_no;\n\
    __global float* that_pos = pos_i + (dim_no * ni);\n\
    float ps_distance = 0;\n\
    float ideal_distance = distances[ (g_i * node_no) + ni ];\n\
    \n\
    for(uint j=0; j < dim_no; ++j){\n\
      coord_dist[j] = p_dim_factors[j] * (this_pos[j] - that_pos[j]);\n\
      //coord_dist[j] = this_pos[j] * (2.0 - this_pos[0]);\n\
      ps_distance += (coord_dist[j] * coord_dist[j]);\n\
    }\n\
\n\
    ps_distance = sqrt(ps_distance);\n\
\n\
    float error = (ideal_distance - ps_distance);\n\
    p_stress += fabs(error);\n\
    if(ps_distance){\n\
      for(uint j=0; j < dim_no; ++j){\n\
	// the coord_dist needs to maintain direction. It's divided by the total\n\
	// euclidean distance to normalise.\n\
	p_fv[j] += (error * coord_dist[j])/ps_distance;\n\
      }\n\
    }\n\
  }\n\
  // and at this point we can create a new position simply as\n\
  float m_factor = 0.5 / (float)node_no;\n\
  for(uint i=0; i < dim_no; ++i)\n\
    p_j[i] = this_pos[i] + (m_factor * p_fv[i]);\n\
  \n\
  stress[g_i] = p_stress;\n\
}";


const char* const move_deltoids_2 = \
"\n\
// runs one iteration of deltoid movement using\n\
// openCL cores\n\
\n\
// modified to use shared (local) memory for pos_i and distances\n\
// to determine if this makes any particular differences. \n\
\n\
// note that for this to work it is necessary to pad the data structures in an appropriate manner\n\
// as we would otherwise be trying to copy memory that hasn't been allocated.\n\
\n\
// pos_i is the current position; pos_j is the next position and will be written\n\
// at the end of the function.\n\
// \n\
// initially we will simply read the positions, do our calculations and \n\
// write to pos_j at the end of the function.\n\
//\n\
// However, it might also be possible to simply switch them depending on an odd / even\n\
// rule to avoid having to run expensive memory copy allocations.\n\
// try something once we know if it's working.\n\
\n\
\n\
// I'm passing the force as a global array that will need to be N * dim long\n\
// This is because it seems that I cannot malloc within the kernel, and I do not\n\
// want to define a maximum dimensionality. If we know the local size, then we\n\
// can make that local, and just make sure that we get the appropriate off-sets\n\
// using get_local_id()\n\
\n\
// a more straightforward approach for allowing dynamic allocation of local memory is\n\
// to make use of 'just in time' compiling by modifying the string. Quite easy to do, by simply\n\
// prepending a set of define statements.\n\
\n\
// This source needs to be prepended with \n\
// #define DIM_NO n\n\
// #define NODE_NO n\n\
// #GLOBAL_ITEM_SIZE n \n\
\n\
__kernel void move_deltoids(__global float* pos_i, __global float* pos_j, __global float* stress, \n\
			    uint dim_no, __global float* dim_factors, __global float* distances, uint node_no,\n\
			    __global unsigned int* error_buffer){\n\
  \n\
  size_t g_i = get_global_id(0);\n\
  size_t l_i = get_local_id(0);\n\
  size_t l_size = get_local_size(0);\n\
\n\
  __local float l_pos_i[DIM_NO * NODE_NO];\n\
  event_t e = async_work_group_copy(l_pos_i, pos_i, DIM_NO * NODE_NO, 0);\n\
\n\
  if(g_i >= node_no)\n\
    return;\n\
\n\
  float p_dim_factors[DIM_NO];\n\
  float p_fv[DIM_NO];\n\
  float coord_dist[DIM_NO];\n\
  float this_pos[DIM_NO];\n\
\n\
\n\
  __global float* p_j = pos_j + (g_i * dim_no);\n\
  float p_stress = 0;\n\
  \n\
  wait_group_events(1, &e);\n\
  // first set the local force_vector to 0\n\
  // can't call any functions so need to iterate through.\n\
  for(uint i=0; i < dim_no; ++i){\n\
    p_fv[i] = 0;\n\
    p_dim_factors[i] = dim_factors[i];\n\
    this_pos[i] = *(l_pos_i + (g_i * dim_no + i));\n\
  }\n\
\n\
  // then go through each node; calculate the current euclidean distance and increment\n\
  // the force_vector as appropriate.\n\
  //__global float* this_pos = pos_i + (g_i * dim_no);\n\
  for(uint i=0; i < node_no; ++i){\n\
    int ni = i;\n\
    __local float* that_pos = l_pos_i + (dim_no * ni);\n\
    float ps_distance = 0;\n\
    float ideal_distance = distances[ (g_i * node_no) + ni ];\n\
    \n\
    for(uint j=0; j < dim_no; ++j){\n\
      coord_dist[j] = p_dim_factors[j] * (this_pos[j] - that_pos[j]);\n\
      ps_distance += (coord_dist[j] * coord_dist[j]);\n\
    }\n\
\n\
    ps_distance = sqrt(ps_distance);\n\
\n\
    float error = (ideal_distance - ps_distance);\n\
    p_stress += fabs(error);\n\
    if(ps_distance){\n\
      for(uint j=0; j < dim_no; ++j){\n\
	// the coord_dist needs to maintain direction. It's divided by the total\n\
	// euclidean distance to normalise.\n\
	p_fv[j] += (error * coord_dist[j])/ps_distance;\n\
      }\n\
    }\n\
  }\n\
  // and at this point we can create a new position simply as\n\
  float m_factor = 0.5 / (float)node_no;\n\
  for(uint i=0; i < dim_no; ++i)\n\
    p_j[i] = this_pos[i] + (m_factor * p_fv[i]);\n\
  \n\
  stress[g_i] = p_stress;\n\
}";


const char* const move_deltoids_dummy = \
"\n\
// dummy kernel. Performs the same number of operations as\n\
// move_deltoids.cl, but does not touch any global memory\n\
// within the major loop.\n\
\n\
// runs one iteration of deltoid movement using\n\
// openCL cores\n\
\n\
// pos_i is the currnet position; pos_j is the next position and will be written\n\
// at the end of the function.\n\
// \n\
// initially we will simply read the positions, do our calculations and \n\
// write to pos_j at the end of the function.\n\
//\n\
// However, it might also be possible to simply switch them depending on an odd / even\n\
// rule to avoid having to run expensive memory copy allocations.\n\
// try something once we know if it's working.\n\
\n\
\n\
// I'm passing the force as a global array that will need to be N * dim long\n\
// This is because it seems that I cannot malloc within the kernel, and I do not\n\
// want to define a maximum dimensionality. If we know the local size, then we\n\
// can make that local, and just make sure that we get the appropriate off-sets\n\
// using get_local_id()\n\
\n\
// a more straightforward approach for allowing dynamic allocation of local memory is\n\
// to make use of 'just in time' compiling by modifying the string. Quite easy to do, by simply\n\
// prepending a set of define statements.\n\
\n\
// This source needs to be prepended with \n\
// #define DIM_NO n\n\
// #define NODE_NO n\n\
\n\
__kernel void move_deltoids(__global float* pos_i, __global float* pos_j, __global float* stress, \n\
			    uint dim_no, __global float* dim_factors, __global float* distances, uint node_no,\n\
			    __global float* force_vector, __global unsigned int* error_buffer){\n\
\n\
  size_t g_i = get_global_id(0);\n\
  size_t l_i = get_local_id(0);\n\
\n\
  if(g_i >= node_no)\n\
    return;\n\
\n\
  float p_dim_factors[DIM_NO];\n\
  float p_fv[DIM_NO];\n\
  float coord_dist[DIM_NO];\n\
  float this_pos[DIM_NO];\n\
\n\
  float p_stress = 0;\n\
  \n\
  // first set the local force_vector to 0\n\
  // these are private rather than local variables, hence no function to\n\
  for(uint i=0; i < dim_no; ++i){\n\
    p_fv[i] = 0;\n\
    p_dim_factors[i] = 1.0; //dim_factors[i];\n\
    this_pos[i] = 0.5; //*(pos_i + (g_i * dim_no + i));\n\
  }\n\
\n\
  // then go through each node; calculate the current euclidean distance and increment\n\
  // the force_vector as appropriate.\n\
  for(uint i=1; i < node_no; ++i){\n\
    // the problem here is that every gpu will want to access the same global variables here\n\
    // compare to 0, and then 1st position. This is likely to be bad. So, rather than that, do\n\
    int ni = (g_i + i) % node_no;\n\
\n\
    float ps_distance = 0;\n\
    float ideal_distance = 1.0; // distances[ (g_i * node_no) + ni ];\n\
    //    float ideal_distance = distances[ (g_i * node_no) + ni ];\n\
    \n\
    for(uint j=0; j < dim_no; ++j){\n\
      coord_dist[j] = 1.0;\n\
      ps_distance += 1.0;\n\
\n\
      //coord_dist[j] = p_dim_factors[j] * (this_pos[j] - that_pos[j]);\n\
      //ps_distance += (coord_dist[j] * coord_dist[j]);\n\
    }\n\
\n\
    ps_distance = sqrt(ps_distance);\n\
\n\
    float error = (ideal_distance - ps_distance);\n\
    p_stress += fabs(error);\n\
    if(ps_distance){\n\
      for(uint j=0; j < dim_no; ++j){\n\
	// the coord_dist needs to maintain direction. It's divided by the total\n\
	// euclidean distance to normalise.\n\
	p_fv[j] += (error * coord_dist[j])/ps_distance;\n\
      }\n\
    }\n\
  }\n\
  // and at this point we can create a new position simply as\n\
  float m_factor = 0.5 / (float)node_no;\n\
  for(uint i=0; i < dim_no; ++i){\n\
    m_factor = 0.5 / (float)node_no;\n\
    //    p_j[i] = this_pos[i]; // + (m_factor * p_fv[i]); // still accesses, but makes no change\n\
  }  \n\
  //  stress[g_i] = p_stress;\n\
}";

#endif
