#include <vector>

using namespace std;

// Versão sequencial
vector<double> jacobi_sequencial(
    vector<vector<double>>& A,
    vector<double>& b,
    int max_iter,
    double epsilon
);

// Versão paralela
vector<double> jacobi_paralelo(
    vector<vector<double>>& A,
    vector<double>& b,
    int max_iter,
    double epsilon,
    int num_threads
);
