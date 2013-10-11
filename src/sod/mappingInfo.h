#ifndef MAPPINGINFO_H
#define MAPPINGINFO_H

#include "stressInfo.h"
#include <vector>
#include <map>

// a way to encapsulate the mapping data
// stress contains the stress and dimensionality
// history. mapped_points the end result.

// WARNING. THIS IS A DANGEROUS STRUCT.
// The destructor does not delete mapped_points
//
// There are two reasons for this.
// 1. I don't want to build in overhead for reference counting
// 2. Deleting the mapped_points would make it difficult
//    to disaggregate the pointer from the struct. Since stress
//    data can actually be rather large, this might be good to do
//    at some point.
struct MappingInfo {
  unsigned int node_no;
  unsigned int dim_no;
  float* mapped_points;
  std::vector<float> node_stress;
  std::vector<stressInfo> stress;
  std::map<std::string, float> timeData;

  MappingInfo()
  {
    node_no = dim_no = 0;
    mapped_points = 0;
  }
  MappingInfo(unsigned int n_no, unsigned int d_no, float* pts, std::vector<stressInfo> s_info)
  {
    node_no = n_no; dim_no = d_no; mapped_points = pts; stress = s_info;
  }

  MappingInfo(unsigned int n_no, unsigned int d_no, float* pts, 
	      std::vector<stressInfo> s_info, std::map<std::string, float> time_data)
  {
    node_no = n_no; dim_no = d_no; mapped_points = pts; stress = s_info;
    timeData = time_data;
  }
  MappingInfo(unsigned int n_no, unsigned int d_no, float* pts, 
	      std::vector<stressInfo> s_info, 
	      std::map<std::string, float> time_data, std::vector<float> n_stress)
  {
    node_no = n_no; dim_no = d_no; mapped_points = pts; stress = s_info;
    timeData = time_data;
    node_stress = n_stress;
  }
  
  ~MappingInfo(){}
};

#endif

