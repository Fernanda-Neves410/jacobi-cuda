#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include "jacobi.h"

__global__ void kernel_jacobi(const double *A, const double *b, 
                            const double *x_old, double *x_new, int n) {
    
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

__global__ void kernel_calcula_erro(const double *x_old, const double *x_new, 
                                    double *erro_global_sq, double *norma_global_sq, 
                                    int n) {
    
    extern __shared__ double sdata[];
    
    double *s_erro = sdata;
    double *s_norma = &sdata[blockDim.x]; 

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        double diff = x_new[i] - x_old[i];
        s_erro[tid] = diff * diff;
        s_norma[tid] = x_new[i] * x_new[i];
    } else {
        s_erro[tid] = 0.0;
        s_norma[tid] = 0.0;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_erro[tid] += s_erro[tid + s];
            s_norma[tid] += s_norma[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(erro_global_sq, s_erro[0]);
        atomicAdd(norma_global_sq, s_norma[0]);
    }
}

void executar_jacobi_cuda(double *h_A, double *h_b, double *h_x_final, 
                            int n, double epsilon, int max_iter) {

    size_t bytes_matriz = n * n * sizeof(double);
    size_t bytes_vetor  = n * sizeof(double);

    double *d_A, *d_b, *d_x_old, *d_x_new, *d_erro, *d_norma;
    cudaMalloc((void**)&d_A, bytes_matriz);
    cudaMalloc((void**)&d_b, bytes_vetor);
    cudaMalloc((void**)&d_x_old, bytes_vetor);
    cudaMalloc((void**)&d_x_new, bytes_vetor);
    cudaMalloc((void**)&d_erro, sizeof(double));
    cudaMalloc((void**)&d_norma, sizeof(double));

    cudaMemset(d_x_old, 0, bytes_vetor); 
    cudaMemset(d_x_new, 0, bytes_vetor);

    cudaMemcpy(d_A, h_A, bytes_matriz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes_vetor, cudaMemcpyHostToDevice);

    int threads_por_bloco = 256;
    int blocos_na_grade = (n + threads_por_bloco - 1) / threads_por_bloco;
    
    size_t shared_mem_size = 2 * threads_por_bloco * sizeof(double);

    int iter = 0;
    double erro_atual = epsilon + 1.0;
    
    double erro_sq_host = 0.0;
    double norma_sq_host = 0.0;

    while (iter < max_iter && erro_atual > epsilon) {
        cudaMemset(d_erro, 0, sizeof(double));
        cudaMemset(d_norma, 0, sizeof(double));

        kernel_jacobi<<<blocos_na_grade, threads_por_bloco>>>(d_A, d_b, d_x_old, 
                                                                d_x_new, n);
        kernel_calcula_erro<<<blocos_na_grade, threads_por_bloco, shared_mem_size>>>
                                                (d_x_old, d_x_new, d_erro, d_norma, n);

        cudaMemcpy(&erro_sq_host, d_erro, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&norma_sq_host, d_norma, sizeof(double), cudaMemcpyDeviceToHost);
        
        erro_atual = sqrt(erro_sq_host) / sqrt(norma_sq_host);

        double *temp = d_x_old;
        d_x_old = d_x_new;
        d_x_new = temp;

        iter++;
    }

    std::cout << "Iteracoes: " << iter << " Erro final: " << erro_atual << "\n";

    cudaMemcpy(h_x_final, d_x_old, bytes_vetor, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_b);
    cudaFree(d_x_old); cudaFree(d_x_new); 
    cudaFree(d_erro); cudaFree(d_norma);
}
