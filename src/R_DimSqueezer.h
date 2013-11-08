#ifndef _R_DimSqueezer
#define _R_DimSqueezer

#include <Rcpp.h>
#include <string>
//#include <math.h>
#include "sod/oCL_DistanceMapperManager.h"  // not necessary if we use a pointer
#include "sod/DistanceMapper.h"

// It seems that this _has_ to be here.
// I could not get away with using it within the RCPP_MODULE part
using namespace Rcpp;   

// standard squeezer
class R_DimSqueezer {
 public :
  R_DimSqueezer(NumericMatrix r_positions);
  ~R_DimSqueezer();

  Rcpp::List squeeze(unsigned int target_dimensionality, unsigned int iter_no);
  Rcpp::List squeezeDF(NumericMatrix dimFactors);
  void useOpenMP(bool use_openMP);
  
 private:
  DistanceMapper* mapper;
  float* positions;
  float* distances;
  unsigned int dimension_no;
  unsigned int node_no;

  bool multithreaded;
};

// openCL based squeezer
// This really ought to be done using inheritance, but .. 
class R_CL_DimSqueezer {
 public:
  R_CL_DimSqueezer(NumericMatrix r_positions);
  ~R_CL_DimSqueezer();

  Rcpp::List squeeze(unsigned int target_dimensionality, unsigned int iter_no, 
		     unsigned int local_work_size);

 private:
  OCL_DistanceMapperManager* mapper;
  float* positions;
  float* distances;
  unsigned int dimension_no;
  unsigned int node_no;
};


RCPP_MODULE(mod_R_DimSqueezer) {
  class_<R_DimSqueezer>("R_DimSqueezer")
    .constructor<NumericMatrix>()
    .method("squeeze", &R_DimSqueezer::squeeze)
    .method("squeezeDF", &R_DimSqueezer::squeezeDF)
    .method("useOpenMP", &R_DimSqueezer::useOpenMP)
    ;

  class_<R_CL_DimSqueezer>("R_CL_DimSqueezer")
    .constructor<NumericMatrix>()
    .method("squeeze", &R_CL_DimSqueezer::squeeze)
    ;
};

#endif
