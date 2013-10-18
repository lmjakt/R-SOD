#include "R_DimSqueezer.h"
#include <vector>
#include <math.h>
#include <string.h> // for memset

// 
float* matrix_to_array(Rcpp::NumericMatrix& mat)
{
  if(!mat.ncol() || !mat.nrow())
    return(0);
  float* array = new float[ mat.ncol() * mat.nrow() ];
  for(unsigned int i=0; i < (unsigned int)mat.nrow(); ++i){
    for(unsigned int j=0; j < (unsigned int)mat.ncol(); ++j){
      array[i * mat.ncol() + j] = (float)mat(i, j);
    }
  }
  return(array);
}

float* inter_node_distances(float* nodes, unsigned int node_no, unsigned int dim_no)
{
  if(!node_no || !dim_no)
    return(0);
  float* dist = new float[ node_no * node_no ];

  for(unsigned int i=0; i < node_no; ++i){
    float* node_i = nodes + i * dim_no;
    for(unsigned int j=0; j < node_no; ++j){
      float* node_j = nodes + j * dim_no;
      float* d = dist + i * node_no + j;
      (*d) = 0;
      for(unsigned int k=0; k < dim_no; ++k)
	(*d) += ((node_i[k] - node_j[k]) * (node_i[k] - node_j[k]));
      (*d) = sqrt(*d);
    }
  }
  return(dist);
}

Rcpp::List prepare_return_data(MappingInfo& mapInfo)
{
  NumericMatrix m_positions(mapInfo.node_no, mapInfo.dim_no);
  
  for(unsigned int i=0; i < mapInfo.node_no; ++i){
    for(unsigned int j=0; j < mapInfo.dim_no; ++j){
      m_positions(i,j) = mapInfo.mapped_points[ i * mapInfo.dim_no + j ]; // This works !?
    }
  }

  NumericMatrix mappingDims(mapInfo.stress.size(), mapInfo.dim_no);
  std::vector<float> globalStress(mapInfo.stress.size());
  for(unsigned int i=0; i < mapInfo.stress.size(); ++i){
    globalStress[i] = mapInfo.stress[i].stress;
    for(unsigned int j=0; j < mapInfo.dim_no; ++j)
      mappingDims(i, j) = mapInfo.stress[i].dimFactors[j];
  }
  
  return(Rcpp::List::create(Rcpp::Named("stress", globalStress),
			    Rcpp::Named("mapDims", mappingDims),
			    Rcpp::Named("pos", m_positions),
			    Rcpp::Named("time", mapInfo.timeData),
			    Rcpp::Named("node_stress", mapInfo.node_stress))	 
	 );
  
}

//////////// Class defintions from here //////////////


////// Ordinary DistanceMapper //////////////
R_DimSqueezer::R_DimSqueezer(Rcpp::NumericMatrix r_positions)
{
  mapper = 0;
  multithreaded = true;
  node_no = r_positions.nrow();
  dimension_no = r_positions.ncol();
  positions = matrix_to_array(r_positions);
  distances = inter_node_distances(positions, node_no, dimension_no);

  if(positions)
    mapper = new DistanceMapper(node_no, dimension_no, positions, distances);
}

R_DimSqueezer::~R_DimSqueezer()
{
  delete []positions;
  delete []distances;
  delete mapper;
}

Rcpp::List R_DimSqueezer::squeeze(unsigned int target_dimensionality, unsigned int iter_no)
{
  if(!mapper)
    return(Rcpp::List::create(Rcpp::Named("Error", "No mapper created")) );
  
  MappingInfo mapInfo = mapper->reduce_dimensions(iter_no, target_dimensionality);
  Rcpp::List return_data = prepare_return_data(mapInfo);
  delete []mapInfo.mapped_points;
  return(return_data);
}

void R_DimSqueezer::useOpenMP(bool use_openMP)
{
  multithreaded = use_openMP;
  if(mapper)
    mapper->use_openMP(use_openMP);
}

////////////// openCL based Distance mapper  ////////////////////
R_CL_DimSqueezer::R_CL_DimSqueezer(Rcpp::NumericMatrix r_positions)
{
  positions = 0;
  distances = 0;
  mapper = 0;
  dimension_no = r_positions.ncol();
  node_no = r_positions.nrow();

  if(!dimension_no || !node_no)
    return;
  
  positions = matrix_to_array(r_positions);
  distances = inter_node_distances(positions, node_no, dimension_no);
  // At this point we can make a mapper and see how that goes
  mapper = new OCL_DistanceMapperManager(node_no, dimension_no, positions, distances);
}

R_CL_DimSqueezer::~R_CL_DimSqueezer()
{
  Rprintf("R_CL_DimSqueezer destructor\n");

  delete []positions;
  delete []distances;
  delete mapper;
}

Rcpp::List R_CL_DimSqueezer::squeeze(unsigned int target_dimensionality, unsigned int iter_no, unsigned int local_work_size)
{
  if(!mapper)
    return(Rcpp::List::create(Rcpp::Named("name", "hello"),
			      Rcpp::Named("data", Rcpp::NumericVector::create(1,2,3))));

  /// This feels a little bit dangerous. We ought perhaps to do some sanity checking of the values that
  /// we are obtaining straight out of R here. But for now, we can always try if we can break the system
  MappingInfo mapInfo = mapper->reduce_dimensions(iter_no, target_dimensionality, local_work_size);

  Rcpp::List return_data = prepare_return_data(mapInfo);
  // We should delete the mapped positions. There is little reason for us to keep them here
  delete []mapInfo.mapped_points;
  return(return_data);
}
