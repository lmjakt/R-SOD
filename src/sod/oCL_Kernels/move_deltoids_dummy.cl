// dummy kernel. Performs the same number of operations as
// move_deltoids.cl, but does not touch any global memory
// within the major loop.

// runs one iteration of deltoid movement using
// openCL cores

// pos_i is the currnet position; pos_j is the next position and will be written
// at the end of the function.
// 
// initially we will simply read the positions, do our calculations and 
// write to pos_j at the end of the function.
//
// However, it might also be possible to simply switch them depending on an odd / even
// rule to avoid having to run expensive memory copy allocations.
// try something once we know if it's working.


// I'm passing the force as a global array that will need to be N * dim long
// This is because it seems that I cannot malloc within the kernel, and I do not
// want to define a maximum dimensionality. If we know the local size, then we
// can make that local, and just make sure that we get the appropriate off-sets
// using get_local_id()

// a more straightforward approach for allowing dynamic allocation of local memory is
// to make use of 'just in time' compiling by modifying the string. Quite easy to do, by simply
// prepending a set of define statements.

// This source needs to be prepended with 
// #define DIM_NO n
// #define NODE_NO n

__kernel void move_deltoids(__global float* pos_i, __global float* pos_j, __global float* stress, 
			    uint dim_no, __global float* dim_factors, __global float* distances, uint node_no,
			    __global float* force_vector, __global unsigned int* error_buffer){

  size_t g_i = get_global_id(0);
  size_t l_i = get_local_id(0);

  if(g_i >= node_no)
    return;

  float p_dim_factors[DIM_NO];
  float p_fv[DIM_NO];
  float coord_dist[DIM_NO];
  float this_pos[DIM_NO];

  //__global float* fv = force_vector + (g_i * dim_no);
  //__global float* p_j = pos_j + (g_i * dim_no);
  float p_stress = 0;
  
  // first set the local force_vector to 0
  // these are private rather than local variables, hence no function to
  for(uint i=0; i < dim_no; ++i){
    p_fv[i] = 0;
    p_dim_factors[i] = 1.0; //dim_factors[i];
    this_pos[i] = 0.5; //*(pos_i + (g_i * dim_no + i));
  }

  // then go through each node; calculate the current euclidean distance and increment
  // the force_vector as appropriate.
  //__global float* this_pos = pos_i + (g_i * dim_no);
  for(uint i=1; i < node_no; ++i){
    // the problem here is that every gpu will want to access the same global variables here
    // compare to 0, and then 1st position. This is likely to be bad. So, rather than that, do
    //int ni = i;
    int ni = (g_i + i) % node_no;

    //    __global float* that_pos = pos_i + (dim_no * ni);

    float ps_distance = 0;
    float ideal_distance = 1.0; // distances[ (g_i * node_no) + ni ];
    //    float ideal_distance = distances[ (g_i * node_no) + ni ];
    
    for(uint j=0; j < dim_no; ++j){
      coord_dist[j] = 1.0;
      ps_distance += 1.0;

      //coord_dist[j] = p_dim_factors[j] * (this_pos[j] - that_pos[j]);
      //ps_distance += (coord_dist[j] * coord_dist[j]);
    }

    ps_distance = sqrt(ps_distance);

    float error = (ideal_distance - ps_distance);
    p_stress += fabs(error);
    if(ps_distance){
      for(uint j=0; j < dim_no; ++j){
	// the coord_dist needs to maintain direction. It's divided by the total
	// euclidean distance to normalise.
	p_fv[j] += (error * coord_dist[j])/ps_distance;
      }
    }
  }
  // and at this point we can create a new position simply as
  float m_factor = 0.5 / (float)node_no;
  for(uint i=0; i < dim_no; ++i){
    m_factor = 0.5 / (float)node_no;
    //    p_j[i] = this_pos[i]; // + (m_factor * p_fv[i]); // still accesses, but makes no change
  }  
  //  stress[g_i] = p_stress;
}
