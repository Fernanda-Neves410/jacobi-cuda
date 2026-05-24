#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <jacobi.h>

using namespace std;


__global__ void kernel_jacobi(const double *A, 
                            const double *b, 
                            const double *x_old, 
                            double *x_new, 
                            int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        double soma = 0.0;

        for (int j = 0; j < n; j++) {
            if (i != j) {
                soma += A[i * n + j] * x_old[j];
            }
        }

        x_new[i] = (b[i] - soma) / A[i * n + i];
    }
}





__global__ void kernel_calcula_erro(const double *x_old, 
                                    const double *x_new, 
                                    double *erro_global_sq, 
                                    int n) {

    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double diff = 0.0;


    if (i < n) {
        diff = x_new[i] - x_old[i];
        sdata[tid] = diff * diff;
    } 
    else {
        sdata[tid] = 0.0;
    }

    __syncthreads();


    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(erro_global_sq, sdata[0]);
    }
}




void executar_jacobi_cuda(double *h_A, 
                        double *h_b, 
                        double *h_x_final, 
                        int n, 
                        double epsilon, 
                        int max_iter) {

    size_t bytes_matriz = n * n * sizeof(double);
    size_t bytes_vetor  = n * sizeof(double);

    double *d_A, *d_b, *d_x_old, *d_x_new, *d_erro;
    cudaMalloc((void**)&d_A, bytes_matriz);
    cudaMalloc((void**)&d_b, bytes_vetor);
    cudaMalloc((void**)&d_x_old, bytes_vetor);
    cudaMalloc((void**)&d_x_new, bytes_vetor);
    cudaMalloc((void**)&d_erro, sizeof(double));

    cudaMemset(d_x_old, 0, bytes_vetor); 
    cudaMemset(d_x_new, 0, bytes_vetor);

    cudaMemcpy(d_A, h_A, bytes_matriz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes_vetor, cudaMemcpyHostToDevice);

    int threads_por_bloco = 256;
  
    int blocos_na_grade = (n + threads_por_bloco - 1) / threads_por_bloco;
    
    size_t shared_mem_size = threads_por_bloco * sizeof(double);

    int iter = 0;
    double erro_atual = epsilon + 1.0;
    double erro_sq_host = 0.0;

    while (iter < max_iter && erro_atual > epsilon) {
        cudaMemset(d_erro, 0, sizeof(double));

        kernel_jacobi<<<blocos_na_grade, threads_por_bloco>>>( d_A, 
                                                            d_b, 
                                                            d_x_old, 
                                                            d_x_new,
                                                            n );

        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess) {
            cout << "Erro no kernel Jacobi: " << cudaGetErrorString(err) << endl;
        }

        cudaDeviceSynchronize();

        kernel_calcula_erro<<< blocos_na_grade, threads_por_bloco, shared_mem_size >>>(
                                                                                d_x_old,
                                                                                d_x_new,
                                                                                d_erro,
                                                                                n );

        err = cudaGetLastError();

        if (err != cudaSuccess) {
            cout << "Erro no kernel erro: " << cudaGetErrorString(err) << endl;
        }

        cudaDeviceSynchronize();
        

        cudaMemcpy(&erro_sq_host, d_erro, sizeof(double), cudaMemcpyDeviceToHost);
        erro_atual = sqrt(erro_sq_host);

        double *temp = d_x_old;
        d_x_old = d_x_new;
        d_x_new = temp;

        iter++;
    }

    cout << "Iteracoes: " << iter << " Erro final: " << erro_atual << "\n";

    cudaMemcpy(h_x_final, d_x_old, bytes_vetor, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_b);
    cudaFree(d_x_old); cudaFree(d_x_new); cudaFree(d_erro);
}
