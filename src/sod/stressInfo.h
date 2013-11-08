#ifndef STRESSINFO_H
#define STRESSINFO_H

#include <vector>

// Note that stressInfo is also defined in distanceMapper.h
// currently in the historical section. We should
struct stressInfo {
  unsigned int dimNo;
  unsigned int activeDimNo;
  std::vector<float> dimFactors;
  float stress;

  stressInfo(){
    dimNo = activeDimNo = 0;
    stress = 0;
  }
  stressInfo(std::vector<float> df, float str)
  : dimNo(df.size()), activeDimNo(df.size()), dimFactors(df), stress(str)
  {
  }
  void setStress(int dim_no, float* df, unsigned int ad, float s){
    dimNo = dim_no;
    activeDimNo = ad;
    dimFactors.resize(dimNo);
    for(unsigned int i=0; i < dimNo; ++i)
      dimFactors[i] = df[i];
    stress = s;
  }
  void setStress(std::vector<float> dims, float s){
    dimNo = activeDimNo = dims.size();
    dimFactors = dims;
    stress = s;
  }
  float dimensionality(){
    float d=0;
    for(unsigned int i=0; i < dimFactors.size(); ++i)
      d += dimFactors[i];
    return(d);
  }
  float currentDF(){
    if(activeDimNo - 1 > 0)
      return(dimFactors[activeDimNo - 1]);
    return(1.0);
  }
};

#endif
