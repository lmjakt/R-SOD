#ifndef DISTANCEMAPPER_H
#define DISTANCEMAPPER_H

// This is a simplified form of the distance mapping algorithm (dimSqueezer)
// that doesn't consider tracking the process.
//
// It also makes use of much simpler data structures, preferring simple C arrays
// to objects representing ideal situations.

// This is written primarily for use as an R extension and as such does not use
// any iostreams, but rather the Rprintf function.

#include <vector>
#include "mappingInfo.h"

// The constructor takes two float arrays, one representing the starting positions of
// objects, and one containing the ideal internode distances. These are not owned by the
// DistanceMapper instance itself and are not released in the destructor.

// There isn't really a strong argument for using a Mapper object here. The best one is
// that the mapper can keep some values (like the force_vectors) that can then be accessed
// by individual functions rather than returned as a complete package. However, that is not
// really that good an argument. 

// a better advantage may be that we can set something like dimFactors in a separate function.

class DistanceMapper
{
 public:
  DistanceMapper(unsigned int node_no, unsigned int dimension_no, float* nodes, float* node_distances);
  ~DistanceMapper();

  MappingInfo reduce_dimensions(std::vector<std::vector<float> >& dimFactorVector);
  MappingInfo reduce_dimensions(unsigned int iter_no, unsigned int target_dim);
  
  void use_openMP(bool useMP);

 private:
  unsigned int node_no;
  unsigned int dimension_no;
  float* nodes;
  float* node_distances;
  bool useOpenMP;

  // variables created by the mapping process
  std::vector<float> dimFactors;
  float* forceVectors;                  // node_no * dimension_no
  float* coordinateVectors;             // node_no * dimension_no
  float* mappedNodes;                   // node_no * dimension_no
  std::vector<float> node_stress;       // node_no
  unsigned int target_dimensionality;

  // mapping functions
  float adjustForces();
  void moveNodes();
  void shrinkDimensionality(unsigned int iter_no);
};

#endif
