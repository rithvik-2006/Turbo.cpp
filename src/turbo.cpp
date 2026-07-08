#include "../include/tensor.hpp"
#include <iostream>
#include <vector>

using namespace std;

int main() {
  cout << "--- Turbo Engine: Tensor Library Test ---\n\n";

  Tensor matrix({1, 2, 3, 4, 5, 6}, {2, 3});
  Tensor bias({10, 20, 30}, {3});

  cout << "Matrix:\n";
  matrix.print();

  Tensor result = matrix + bias;
  cout << "\nMatrix + Bias (Broadcasting):\n";
  result.print();

  cout << "\nTransposed Result (Zero-Copy):\n";
  result.transpose().print();

  return 0;
}