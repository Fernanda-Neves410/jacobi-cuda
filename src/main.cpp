#include "jacobi.h"
#include "utils.h"
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

extern void executar_jacobi_cuda( double *h_A, double *h_b, double *h_x_final,
                                 int n, double tol, int max_iter);


int main() {

    double epsilon = 1e-6;
    int max_iter = 5000;



    // alterar tamanho da matriz
    int n = 100;

    // Cria sistema linear
    vector<vector<double>> A(n, vector<double>(n, 0.0));
    vector<double> b(n, 0.0);

    gerar_matriz(A, b, n);




    // Execução Sequencial
    auto inicio_seq = chrono::high_resolution_clock::now();

    auto resultado = jacobi_sequencial(A, b, max_iter, epsilon);

    auto fim_seq = chrono::high_resolution_clock::now();

    chrono::duration<double> tempo_gasto_seq = fim_seq - inicio_seq;

    double temp_total = tempo_gasto_seq.count();

    cout << "Tempo Sequencial: " << temp_total << "s\n\n";




    // Preparação CUDA
    double *h_A = new double[n * n];
    double *h_b = new double[n];
    double *h_x_final = new double[n];

    converter_matriz_para_vetor(A, b, h_A, h_b, n);




    // Execução CUDA
    auto inicio_cuda = chrono::high_resolution_clock::now();

    executar_jacobi_cuda(h_A, h_b, h_x_final, n, epsilon, max_iter);

    auto fim_cuda = chrono::high_resolution_clock::now();

    chrono::duration<double> tempo_gasto_cuda = fim_cuda - inicio_cuda;

    double tempo_cuda = tempo_gasto_cuda.count();

    cout << "Tempo CUDA: " << tempo_cuda << "s\n\n";





    // Speedup
    double speedup = temp_total / tempo_cuda;

    cout << "Speedup: " << speedup << endl;




    // Diferença CPU vs GPU
    double diferenca = calcular_diferenca(resultado, h_x_final, n);

    cout << "Diferenca CPU vs GPU: " << diferenca << endl;




    delete[] h_A;
    delete[] h_b;
    delete[] h_x_final;

    return 0;
}
