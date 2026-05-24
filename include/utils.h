#pragma once

#include <vector>

using namespace std;

// Gera uma matriz diagonal dominante
void gerar_sistema(
    vector<vector<double>>& A,
    vector<double>& b,
    int n
);

// Converte matriz 2D para vetor 1D
void converter_matriz_para_vetor(
    vector<vector<double>>& A,
    double *h_A
);

// Calcula diferença entre resultado CPU e GPU
double calcular_diferenca(
    vector<double>& resultado_cpu,
    double *resultado_gpu,
    int n
);