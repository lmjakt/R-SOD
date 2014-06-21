#include <Rcpp.h>
#include "DistanceMapper.h"
#include <iostream>   // but lose this later

//bool useOpenMP = true;

DistanceMapper::DistanceMapper(unsigned int node_no, unsigned int dimension_no, float* nodes, float* node_distances)
  : node_no(node_no), dimension_no(dimension_no), nodes(nodes), node_distances(node_distances), useOpenMP(true)
{
  forceVectors = 0;
  mappedNodes = 0;
  coordinateVectors = 0;
  if(node_no && dimension_no){
    dimFactors.resize(dimension_no);
    dimFactors.assign(dimension_no, 1.0);
    forceVectors = new float[ node_no * dimension_no ];
    memset((void*)forceVectors, 0, sizeof(float) * node_no * dimension_no);
    mappedNodes = new float[ node_no * dimension_no ];
    memcpy((void*)mappedNodes, (void*)nodes, sizeof(float) * node_no * dimension_no);

    coordinateVectors = new float[ node_no * dimension_no ];
    node_stress.resize(node_no);
  }else{
    Rprintf("Expect to crash, memory not inited, node_no: %d dimension_no %d\n", node_no, dimension_no);
  }
}

DistanceMapper::~DistanceMapper()
{
  delete []forceVectors;
  delete []mappedNodes;  
  delete []coordinateVectors;
}

MappingInfo DistanceMapper::reduce_dimensions(unsigned int iter_no, unsigned int target_dim)
{
  MappingInfo dummyInfo;
  if(dimension_no < 2){
    Rprintf("Starting dimensionality is 1. This is pointless\n");
    return(dummyInfo);
  }
  if(target_dim < dimension_no){
    target_dimensionality = target_dim;
  }else{
    Rprintf("Target dimensionality %d is not smaller than starting dimensionality %d\ntarget_dim defaulting to 2\n",
	    target_dim, dimension_no);
    target_dimensionality = 2;
  }
  if(iter_no < 1){
    Rprintf("Error: reduce_dimensions requested 0 iterations\n");
    return(dummyInfo);
  }
  dimFactors.assign(dimension_no, 1.0);
  std::vector<std::vector<float> > dimFactorVector(iter_no);
  // This is a bit stupid, and should be changed // 
  for(unsigned int i=0; i < dimFactorVector.size(); ++i){
    shrinkDimensionality(iter_no);
    dimFactorVector[i] = dimFactors;
  }
  return(reduce_dimensions(dimFactorVector));
}

MappingInfo DistanceMapper::reduce_dimensions(std::vector<std::vector<float> >& dimFactorVector)
{
  
  MappingInfo dummyInfo;
  if(!nodes || !node_distances || !forceVectors || !mappedNodes)
    return(dummyInfo);
  // we have to verify that the dimFactorVector is reasonable
  // Each sub vector should be of dimensionality length and
  // each value should be between 1 and 0, though actually
  // one might be able to create interesting effects by using
  // different values.
  if(dimFactorVector.size() == 0){
    Rprintf("Error: reduce_dimensions no iterations specified\n");
    return(dummyInfo);
  }
  unsigned int df_size = dimFactorVector[0].size();
  for(unsigned int i=0; i < dimFactorVector.size(); ++i){
    if(dimFactorVector[i].size() != df_size){
      Rprintf("Incorrect DimFactorVector specified iteration %d has %d dimensions\n",
	      i, dimFactorVector[i].size());
      Rprintf("Using default dimension strategy\n");
      return( reduce_dimensions(dimFactorVector.size(), 2) ); // potential infinite loop
    }
  }

  unsigned int iter_no = dimFactorVector.size();
  std::vector<stressInfo> stress_data(iter_no);

  // make sure that the mappedNodes are set to the original ones.
  // This should be optional, as we may want to keep the nodes for a longer time.
  memcpy((void*)mappedNodes, (void*)nodes, sizeof(float) * node_no * dimension_no);
  
  Rprintf("Squeezing\n");
  // Let's print out 80 colums to indicate progress
  unsigned int columns = 80;
  unsigned int div = (iter_no / columns);
  div = (div == 0) ? 1 : div;
  for(unsigned int i=0; i < iter_no; ++i){
    if(!(i % div)) Rprintf("-");
  }
  Rprintf("|\n");
  
  for(unsigned int i=0; i < iter_no; ++i){
    // progress bar
    if(!(i % div))
      Rprintf("=");
    dimFactors = dimFactorVector[i];  // this is a hack to allow setting of dimFactors
    float stress = adjustForces();
    stress_data[i].setStress(dimFactors, stress);
    
    moveNodes();
  }
  Rprintf("\n");
  // Reduce stress while it is decreasing, or we reach a max iteration no.
  float last_stress = stress_data.back().stress;
  bool remove_residual_stress = true;
  while(remove_residual_stress && stress_data.size() < (3 * iter_no)){
    Rprintf(".");
    float stress = adjustForces();
    moveNodes();
    //    Rprintf("%f\t%f\n", last_stress, stress);
    stress_data.push_back( stressInfo(dimFactors, stress) );
    remove_residual_stress = (stress < last_stress);
    last_stress = stress;
  }
  Rprintf("\n");

  // prepare the mapping info in a reasonable manner
  std::map<std::string, float> time_data;
  time_data["Undefined"] = 0;

  // make a copy of the mappedNodes
  float* node_copy = new float[ node_no * dimension_no ];
  memcpy((void*)node_copy, (void*)mappedNodes, sizeof(float) * node_no * dimension_no);

  return( MappingInfo(node_no, dimension_no, node_copy, stress_data, time_data, node_stress) ); 
}

void DistanceMapper::use_openMP(bool useMP)
{
  useOpenMP = useMP;
}

float DistanceMapper::adjustForces()
{
  memset((void*)forceVectors, 0, sizeof(float) * node_no * dimension_no);
  float stress = 0;
  // for each point we need a coordinate vector giving the direction of differences.
  // This is a bit problematic if we want to parallelise using openMP, using simple
  // pragma statements. Although somewhat wasteful of memory, using pre-assigned
  // memory for the whole lot is the simplest solution.

  // To speed this up we can pre-allocate the memory in the constructor
  
  unsigned int dim_no = dimFactors.size() < dimension_no ? dimFactors.size() : dimension_no;
  // this is wasteful as we have already assigned the force vectors;
  // but since dimension_no is used elsewhere we have to be a little bit careful with
  // changing the forceVector allocation.

  #pragma omp parallel for if (useOpenMP)
  for(unsigned int i=0; i < node_no; ++i){
    float* cv = coordinateVectors + i * dimension_no;
    float* fv = forceVectors + i * dimension_no;
    float* node_i = mappedNodes + i * dimension_no;

    for(unsigned int j=0; j < node_no; ++j){
      float* node_j = mappedNodes + j * dimension_no;
      memset((void*)cv, 0, sizeof(float) * dimension_no);
      float f_distance = node_distances[ i * node_no + j ]; // the full distance
      float r_distance = 0;                                 // distance in reduced space
      for(unsigned int k=0; k < dim_no; ++k){
	cv[k] = dimFactors[k] * (node_i[k] - node_j[k]);
	r_distance += (cv[k] * cv[k]);
      }
      r_distance = sqrt(r_distance);
      float error = (f_distance - r_distance);
      node_stress[i] = fabs(error);  // sign kept in the openCL version left for compatibility for now.
      
      //      #pragma omp atomic  // this slows things massively!
      //stress += node_stress[i];

      // if r_distance is = 0 we will get a divide by 0 error.
      // int this case we cannot set a direction for the vector, 
      // and are better off not setting it at all.
      if(r_distance > 0){
	for(unsigned int k=0; k < dimension_no; ++k)
	  fv[k] += (error * cv[k])/r_distance;
      }
    }
  }
  for(unsigned int i=0; i < node_no; ++i)
    stress += node_stress[i];
  return(stress);
}

void DistanceMapper::moveNodes()
{
  float move_factor = 0.5 / node_no;
  for(unsigned int i=0; i < node_no; ++i){
    float* node = mappedNodes + i * dimension_no;
    float* fv = forceVectors + i * dimension_no;
    for(unsigned k=0; k < dimension_no; ++k)
      node[k] += (move_factor * fv[k]);
  }
}

void DistanceMapper::shrinkDimensionality(unsigned int iter_no)
{
  float r_factor = (1.0 / 0.9) / (float)iter_no;
  for(unsigned int i=target_dimensionality; i < dimension_no; ++i)
    dimFactors[i] = (dimFactors[i] - r_factor) < 0 ? 0 : (dimFactors[i] - r_factor);
}
