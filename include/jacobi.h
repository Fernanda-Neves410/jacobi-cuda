#pragma once

#include <vector>

using namespace std;

// Versão sequencial
vector<double> jacobi_sequencial(
    vector<vector<double>>& A,
    vector<double>& b,
    int max_iter,
    double epsilon
);

// Versão paralela com CUDA
// blockSize servirá para dizer qual o tamanho do bloco
// que estamos trabalhando na GPU
vector<double> jacobi_cuda(
    vector<vector<double>>& A,
    vector<double>& b,
    int max_iter,
    double epsilon,
    int blockSize
);