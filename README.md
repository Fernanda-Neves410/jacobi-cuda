# Método de Jacobi com CUDA

Implementação do método iterativo de Jacobi utilizando CUDA para resolução paralela de sistemas lineares em GPU.

---

# Sobre o Projeto

Este projeto foi desenvolvido para a disciplina de Programação Paralela com o objetivo de implementar e analisar o desempenho do método iterativo de Jacobi utilizando processamento paralelo em GPU através da API CUDA.

O trabalho compara duas abordagens:

- Implementação sequencial executada na CPU;
- Implementação paralela executada em GPU utilizando CUDA.

A proposta explora conceitos importantes de computação paralela, como:

- paralelismo massivo;
- organização de threads em blocos e grades;
- memória compartilhada;
- sincronização;
- análise de desempenho e speedup.

---

# Método de Jacobi

O método de Jacobi é um algoritmo iterativo utilizado para resolver sistemas lineares da forma:

```
Ax = b
```

Cada elemento do vetor solução é calculado independentemente utilizando apenas os valores da iteração anterior, tornando o algoritmo altamente paralelizável.

A atualização de cada posição é dada por:

```
x_i^{(k+1)} = \frac{1}{a_{ii}} \left(b_i - \sum_{j \neq i} a_{ij}x_j^{(k)}\right)
```

---

# Estrutura do Projeto

```text
.
├── main.cpp              # Execução principal e medições de desempenho
├── sequencial.cpp        # Implementação sequencial
├── cuda.cu               # Implementação CUDA e kernels da GPU
├── jacobi.h              # Assinaturas das funções principais
├── utils.cpp             # Funções auxiliares
├── utils.h               # Assinaturas das funções utilitárias
├── README.md
└── relatorio.pdf
```

---

# Implementação CUDA

A implementação paralela utiliza dois kernels principais:

## kernel_jacobi

Responsável pelo cálculo das novas aproximações do método de Jacobi.

Cada thread calcula uma equação do sistema linear.

---

## kernel_calcula_erro

Responsável pelo cálculo do erro relativo entre iterações consecutivas.

A redução paralela é realizada utilizando memória compartilhada (`shared memory`) e operações atômicas (`atomicAdd`).

---

# Organização das Threads

A execução na GPU foi organizada utilizando:

- Threads por bloco:
  - 64
  - 128
  - 256
  - 512
  - 1024

O número de blocos é calculado dinamicamente de acordo com o tamanho do sistema linear.

---

# Critério de Parada

O algoritmo encerra quando:

```
\frac{||x^{(k+1)} - x^{(k)}||}{||x^{(k+1)}||} < \epsilon
```

ou quando o número máximo de iterações é atingido.

---

# Tecnologias Utilizadas

- C++
- CUDA
- NVIDIA CUDA Toolkit
- NVCC
- GPU NVIDIA RTX 3060

---

# Compilação

Exemplo utilizando NVCC:

```bash
nvcc main.cpp sequencial.cpp cuda.cu utils.cpp -o jacobi
```

---

# Execução

```bash
./jacobi
```

---

# Resultados Obtidos

O programa realiza:

- medição do tempo sequencial;
- medição do tempo em GPU;
- cálculo de speedup;
- comparação entre os resultados CPU e GPU;
- análise de convergência.

---

# Possíveis Melhorias

- utilização de memória constante;
- otimização de acessos globais;
- redução paralela mais eficiente;
- suporte a matrizes maiores;
- múltiplas GPUs.

---

# Autores

- Fernanda das Neves Merqueades Santos
- Kauê Peixoto

---

# Disciplina

Programação Paralela com CUDA  
Universidade Federal de Mato Grosso do Sul