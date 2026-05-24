#pragma once

#include <vector>

using namespace std;

// Versão sequencial
vector<double> jacobi_sequencial(vector<vector<double>>& A, vector<double>& b,
                                int max_iter, double epsilon);


// Versão paralela com CUDA
vector<double> executar_jacobi_cuda(vector<vector<double>>& A, vector<double>& b,
                                    int n, double epsilon, int max_iter);