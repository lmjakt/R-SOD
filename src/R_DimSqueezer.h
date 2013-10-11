#ifndef _R_DimSqueezer
#define _R_DimSqueezer

#include <Rcpp.h>
#include <string>
#include <math.h>
#include "sod/oCL_DistanceMapperManager.h"  // not necessary if we use a pointer

// It seems that this _has_ to be here.
// I could not get away with using it within the RCPP_MODULE part
using namespace Rcpp;   

class R_DimSqueezer {
 public:
  R_DimSqueezer(NumericMatrix r_positions);
  ~R_DimSqueezer();

  Rcpp::List squeeze(unsigned int target_dimensionality, unsigned int iter_no, 
		     unsigned int local_work_size);

 private:
  OCL_DistanceMapperManager* mapper;
  float* positions;
  float* distances;
  unsigned int dimension_no;
  unsigned int node_no;

  // an inline function for calculating a distance
  float e_dist(float* n1, float* n2){
    float d = 0;
    for(unsigned int i=0; i < dimension_no; ++i)
      d += (n1[i] - n2[i]) * (n1[i] - n2[i]);
    return(sqrt(d));
  }

};

RCPP_MODULE(mod_R_DimSqueezer) {
  class_<R_DimSqueezer>("R_DimSqueezer")

    .constructor<NumericMatrix>()

    .method("squeeze", &R_DimSqueezer::squeeze)
    ;
};

#endif
