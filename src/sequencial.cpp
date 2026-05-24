#include "jacobi.h"
#include <cmath>
#include <iostream>

using namespace std;

vector<double> jacobi_sequencial( vector<vector<double>>& A, vector<double>& b,
                                int max_iter, double epsilon) {
                                    
    int n = A.size();

    vector<double> x_old(n, 0.0);
    vector<double> x_new(n, 0.0);

    for (int k = 0; k < max_iter; k++) {

        // Calcula nova iteração
        for (int i = 0; i < n; i++) {
            double soma = 0.0;

            for (int j = 0; j < n; j++) {
                if (j != i) {
                    soma += A[i][j] * x_old[j];
                }
            }

            x_new[i] = (b[i] - soma) / A[i][i];
        }

        // Calcula erro
        double erro = 0.0;
        double norma = 0.0;

        for (int i = 0; i < n; i++) {

            erro += pow(x_new[i] - x_old[i], 2);

            norma += pow(x_new[i], 2);
        }

        erro = sqrt(erro) / sqrt(norma);

        // Verifica convergência
        if (erro < epsilon) {
            cout << "Convergiu em "
                << k + 1
                << " iteracoes.\n";

            break;
        }

        x_old = x_new;
    }

    return x_new;
}