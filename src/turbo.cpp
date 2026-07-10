#include "../include/turbo/tensor/tensor.hpp"
#include <iostream>
#include <vector>
using namespace std;

int main() {
  cout << "--- Turbo Engine: Cache-Friendly MatMul Test ---\n\n";

  // Create a 2x3 matrix
  Tensor A({1, 2, 3, 4, 5, 6}, {2, 3});

  // Create a 3x2 matrix
  Tensor B({7, 8, 9, 10, 11, 12}, {3, 2});

  cout << "Matrix A (2x3):\n";
  A.print();

  cout << "\nMatrix B (3x2):\n";
  B.print();

  // Execute the compute engine
  Tensor C = A.matmul(B);

  cout << "\nResult Matrix C (2x2) [A * B]:\n";
  C.print();

  return 0;
}