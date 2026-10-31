#include "../include/cudadnnUtil.cuh"
#include <cuda.h>
#include <cstdio>
#include <stdlib.h>
#include <time.h>

// Error checking marco
#define CUDA_CHECK_RET(x)                                               \
  do {                                                                  \
    cudaError_t err_code = x;                                           \
    if (err_code != 0) {                                                \
      printf("Failed at %s:%d w/ code %d <%s>, abort\n",                \
          __FILE__, __LINE__, err_code, cudaGetErrorString(err_code));  \
      exit(1);                                                          \
    }                                                                   \
  } while (0)

// Workspace
float *workspace_base;

// helper functions
void generate_data(float *base, long long size) {
  for(long long i = 0; i < size; i++){
    base[i] = (float)(rand() % 256); 
  }
}

void generate_mask(float *base, long long size, float p) {
  for(long long i = 0; i < size; i++){
    float m = (float) (rand() % 1000000); 
    base[i] = m < (p * 1000000) ? 1.0f : 0.0f;
  }
}

// declare all tensors
float *tensor808, *tensor822, *tensor823, *tensor825, *tensor827, *tensor824;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // Conv2d Size: 294912
  apply_grad<<<64, 0>>>(tensor823, tensor824, 294912);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("199 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 130809856);
  
  // Weight addr assignment
  tensor808 = workspace_base + 0;
  tensor822 = workspace_base + 12845056;
  tensor823 = workspace_base + 16056320;
  tensor825 = workspace_base + 16351232;
  tensor827 = workspace_base + 29196288;
  tensor824 = workspace_base + 32407552;

  // Data & mask generation
  generate_data(tensor808, 12845056);
  generate_data(tensor822, 3211264);
  generate_data(tensor823, 294912);
  generate_data(tensor825, 12845056);
  generate_data(tensor827, 3211264);
  generate_data(tensor824, 294912);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
