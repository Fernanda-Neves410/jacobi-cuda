#include "jacobi.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <chrono>

extern void executar_jacobi_cuda(double *h_A, double *h_b, double *h_x_final, int n, double tol, int max_iter);

int main() {
    double epsilon = 1e-6;
    int max_iter = 5000;
    int i, j;
    double temp_inicial, temp_final, temp_total, 
            speedup, eficiencia;

    // alterar tamanho da matriz
    int n = 1000;

    vector<vector<double>> A(n, vector<double>(n, 0.0));
    vector<double> b(n, 0.0);

    // Gera uma matriz diagonal dominante
    // Cria todos os elementos da matriz com valor 1.0
    // Add (soma de todos os elementos da linha + 1.0) nos elementos da diagonal principal
    // valores de b = {2, 4, 8, 10, ...)
    for (i = 0; i < n; i++) {
            double soma_linha = 0.0;
            for (j = 0; j < n; j++) {
                if (i != j) {
                    A[i][j] = 1.0; 
                    soma_linha += A[i][j];
                }
            }
            A[i][i] = soma_linha + 1.0; 
            b[i] = (i + 1) * 2.0; 
    }

    //  sequencial
    temp_inicial = omp_get_wtime();
    auto resultado = jacobi_sequencial(A, b, max_iter, epsilon);
    temp_final = omp_get_wtime();
    temp_total = temp_final - temp_inicial;
    double temp_seq = temp_total;
    cout << "Tempo sequencial: " << temp_seq << "s\n\n";

    double *h_A = new double[n * n];
    double *h_b = new double[n];
    double *h_x_final = new double[n];

    // inicia o host de A e b
    // Obs: transforma a matriz em um vetor
    for (int linha = 0; linha < n; linha++) {
        h_b[linha] = b[linha];
        for (int coluna = 0; coluna < n; coluna++) {
            h_A[linha * n + coluna] = A[linha][coluna];
        }
    }

    // executa e calcula o tempo do jacobi_cuda
    auto inicio = std::chrono::high_resolution_clock::now();
    executar_jacobi_cuda(h_A, h_b, h_x_final, n, epsilon, max_iter);
    auto fim = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tempo_gasto = fim - inicio;
    

    return 0;
}
