#ifndef OCL_DISTANCEMAPPERMANAGER_H
#define OCL_DISTANCEMAPPERMANAGER_H

#include <string>
#include "mappingInfo.h"

class node_set;
class OCL_DistanceMapper;

class OCL_DistanceMapperManager {
 public:
  OCL_DistanceMapperManager(unsigned int node_no, unsigned int dim_no, float* nodes, float* node_dist);
  OCL_DistanceMapperManager(node_set* pos, node_set* dist);
  ~OCL_DistanceMapperManager();
  
  // testing function, returns nothing..always starts with the full positions
  // in N-dimensional space. Get the function to print out some timing info
  MappingInfo reduce_dimensions(unsigned int iter_no, unsigned int target_dim, unsigned int local_work_size);

  void print_pointers();

 private:
  unsigned int node_no;
  unsigned int dimensionality;
  float* nodes;
  float* node_distances;

  OCL_DistanceMapper* mapper;

  std::string cl_define_statement();
};

#endif
