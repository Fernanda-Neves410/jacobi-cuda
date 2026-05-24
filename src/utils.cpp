#include "utils.h"
#include <cmath>

void gerar_matriz(vector<vector<double>>& A, vector<double>& b, int n) {

    for (int i = 0; i < n; i++) {

        double soma_linha = 0.0;

        for (int j = 0; j < n; j++) {

            if (i != j) {
                A[i][j] = 1.0;
                soma_linha += A[i][j];
            }
        }

        A[i][i] = soma_linha + 1.0;

        b[i] = (i + 1) * 2.0;
    }
}


void converter_matriz_para_vetor(vector<vector<double>>& A, vector<double>& b,
                                double *h_A, double *h_b, int n) {

    for (int linha = 0; linha < n; linha++) {

        h_b[linha] = b[linha];

        for (int coluna = 0; coluna < n; coluna++) {

            h_A[linha * n + coluna] =
                A[linha][coluna];
        }
    }
}


double calcular_diferenca(vector<double>& resultado_cpu, double *resultado_gpu, int n) {

    double diferenca = 0.0;

    for (int i = 0; i < n; i++) {

        diferenca += abs(
            resultado_cpu[i] - resultado_gpu[i]
        );
    }

    return diferenca;
}