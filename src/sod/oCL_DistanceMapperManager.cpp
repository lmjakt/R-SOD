#include "oCL_DistanceMapperManager.h"
#include "node_set.h"
#include "oCL_DistanceMapper.h"
#include <iostream>
#include <sstream>
#include <Rcpp.h>

typedef unsigned int uint;

OCL_DistanceMapperManager::OCL_DistanceMapperManager(unsigned int node_no, unsigned int dim_no, float* nodes, float* node_dist)
  : node_no(node_no), dimensionality(dim_no), nodes(nodes), node_distances(node_dist)
{
  mapper = 0;
  if(!node_no || !dimensionality)
    return;
  // and take a lot of things on fate here...
  mapper = new OCL_DistanceMapper(cl_define_statement());
  mapper->device_properties();
  mapper->kernel_properties();
}

OCL_DistanceMapperManager::OCL_DistanceMapperManager(node_set* pos, node_set* dist)
{
  nodes = node_distances = 0;
  node_no = pos->n_size();
  dimensionality = pos->n_dim();
  mapper = 0;
  if(!node_no || !dimensionality){
    Rprintf("OCL_DistanceMapperManager empty positions %d * %d\n", 
	    pos->n_size(), pos->n_dim());
    return;
  }
  
  if(dist->n_size() != node_no || dist->n_dim() != (node_no)){
    Rprintf("OCL_DistanceMapperManager dist has unsuitable dimensions node_no %d dist_dim %d\n",
	    node_no, dist->n_dim());
    return;
  }
  
  nodes = new float[node_no * dimensionality];
  node_distances = new float[node_no * node_no];

  for(uint i=0; i < node_no; ++i){
    for(uint j=0; j < dimensionality; ++j)
      nodes[ i * dimensionality + j] = pos->value(i, j);
    for(uint j=0; j < node_no; ++j)
      node_distances[i * node_no + j] = dist->value(i, j);
  }

  mapper = new OCL_DistanceMapper(cl_define_statement());
  mapper->device_properties();
  mapper->kernel_properties();
}

OCL_DistanceMapperManager::~OCL_DistanceMapperManager()
{
  Rprintf("OCL_DistanceMapperManager destructor invoked\n");
  delete mapper;
  //delete []nodes;
  //delete []node_distances; // these are deleted by the parent
}


MappingInfo OCL_DistanceMapperManager::reduce_dimensions(unsigned int iter_no, unsigned int target_dim, 
						  unsigned int local_work_size)
{
  mapper->set_local_item_size(local_work_size);

  MappingInfo mapInfo = mapper->reduce_dimensions(nodes, node_no, dimensionality, target_dim,
					  iter_no, node_distances);
  return(mapInfo);
  // and this we should change so that it reg
}

void OCL_DistanceMapperManager::print_pointers()
{
  Rprintf("nodes: %p\n", nodes);
  Rprintf("node_distances: %p\n", node_distances);
}

std::string OCL_DistanceMapperManager::cl_define_statement()
{
  std::ostringstream define_statement;
  define_statement << "#define DIM_NO " << dimensionality << "\n"
		   << "#define NODE_NO " << node_no;
  return( define_statement.str());
}
