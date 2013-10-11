#include "R_DimSqueezer.h"
#include <vector>
#include <math.h>

R_DimSqueezer::R_DimSqueezer(Rcpp::NumericMatrix r_positions)
{
  positions = 0;
  distances = 0;
  mapper = 0;
  dimension_no = r_positions.ncol();
  node_no = r_positions.nrow();

  if(!dimension_no || !node_no)
    return;
  
  positions = new float[ dimension_no * node_no ];
  distances = new float[ node_no * node_no ];
  memset((void*)distances, 0, sizeof(float) * node_no * node_no);
  // This seems rather horrid, but my reading of the docs isn't good enough
  // to find a better way.
  for(unsigned int i=0; i < node_no; ++i){
    for(unsigned int j=0; j < dimension_no; ++j){
      positions[ i * dimension_no + j ] = (float)r_positions(i, j);
    }
  }
  // Create a set of distances. Somewhat redundant, but simple.
  for(unsigned int i=0; i < node_no; ++i){
    for(unsigned int j=0; j < node_no; ++j){
      distances[ i * node_no + j ] = e_dist( positions + i * dimension_no, 
					     positions + j * dimension_no );
    }
  }
  // At this point we can make a mapper and see how that goes
  mapper = new OCL_DistanceMapperManager(node_no, dimension_no, positions, distances);
}

R_DimSqueezer::~R_DimSqueezer()
{
  Rprintf("R_DimSqueezer destructor\n");

  delete []positions;
  delete []distances;
  delete mapper;
}

Rcpp::List R_DimSqueezer::squeeze(unsigned int target_dimensionality, unsigned int iter_no, unsigned int local_work_size)
{
  if(!mapper)
    return(Rcpp::List::create(Rcpp::Named("name", "hello"),
			      Rcpp::Named("data", Rcpp::NumericVector::create(1,2,3))));

  /// This feels a little bit dangerous. We ought perhaps to do some sanity checking of the values that
  /// we are obtaining straight out of R here. But for now, we can always try if we can break the system
  MappingInfo mapInfo = mapper->reduce_dimensions(iter_no, target_dimensionality, local_work_size);
  NumericMatrix m_positions(node_no, dimension_no);

  for(unsigned int i=0; i < node_no; ++i){
    for(unsigned int j=0; j < dimension_no; ++j){
      m_positions(i,j) = mapInfo.mapped_points[ i * dimension_no + j ]; // This works !?
    }
  }

  NumericMatrix mappingDims(mapInfo.stress.size(), dimension_no);
  std::vector<float> globalStress(mapInfo.stress.size());
  for(unsigned int i=0; i < mapInfo.stress.size(); ++i){
    globalStress[i] = mapInfo.stress[i].stress;
    for(unsigned int j=0; j < dimension_no; ++j)
      mappingDims(i, j) = mapInfo.stress[i].dimFactors[j];
  }
  
  return(Rcpp::List::create(Rcpp::Named("stress", globalStress),
			    Rcpp::Named("mapDims", mappingDims),
			    Rcpp::Named("pos", m_positions),
			    Rcpp::Named("time", mapInfo.timeData),
			    Rcpp::Named("node_stress", mapInfo.node_stress))	 
	 );
  // We should delete the mapped positions. There is little reason for us to keep them here
  delete []mapInfo.mapped_points;
}
