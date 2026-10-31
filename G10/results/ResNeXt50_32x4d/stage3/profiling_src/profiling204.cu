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
float *tensor803, *tensor804, *tensor805, *tensor807, *tensor809, *tensor806;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // Conv2d Size: 1048576
  apply_grad<<<64, 0>>>(tensor805, tensor806, 1048576);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("204 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 213909504);
  
  // Weight addr assignment
  tensor803 = workspace_base + 0;
  tensor804 = workspace_base + 12845056;
  tensor805 = workspace_base + 25690112;
  tensor807 = workspace_base + 26738688;
  tensor809 = workspace_base + 39583744;
  tensor806 = workspace_base + 52428800;

  // Data & mask generation
  generate_data(tensor803, 12845056);
  generate_data(tensor804, 12845056);
  generate_data(tensor805, 1048576);
  generate_data(tensor807, 12845056);
  generate_data(tensor809, 12845056);
  generate_data(tensor806, 1048576);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
