#ifndef NODE_SET_H
#define NODE_SET_H

#include <string>
#include <vector>

class node_set {
 public:
  node_set();
  node_set(std::vector<std::string> labels, std::vector<std::vector<float> > nodes);
  node_set(std::vector<std::string> names, std::vector<std::vector<float> > nodes, std::vector<std::string> col_labels);
  
  node_set distances();

  unsigned int n_size();
  unsigned int n_dim();
  std::vector<std::vector<float> > Nodes();
  std::vector<float> node(unsigned int n, std::string* label=0);
  float value(unsigned int n, unsigned int d, std::string* label=0);
  std::vector<std::string> Labels();
  std::vector<std::string> Col_labels();

  bool push_node(std::string label, std::vector<float> v);
  bool set_node(unsigned int n, std::vector<float> v);
  bool set_col_header(std::vector<std::string>& ch);
  
 private:
  void init();
  float e_distance(std::vector<float>& a, std::vector<float>& b);

  std::vector<std::string> labels;
  std::vector<std::vector<float> > nodes;
  std::vector<std::string> col_labels;
  unsigned int dimensions;
};


#endif
