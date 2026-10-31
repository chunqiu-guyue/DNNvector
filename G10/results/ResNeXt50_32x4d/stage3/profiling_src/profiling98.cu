#include "../include/declaration1.cuh"
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
float *tensor635, *tensor691, *tensor695, *tensor676, *tensor751;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // NumInputs:2, Size:12845056
  forwardPass_Add_l97<<<64, 0>>>(tensor635, tensor676, tensor691, 12845056);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("098 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 256901120);
  
  // Weight addr assignment
  tensor635 = workspace_base + 0;
  tensor691 = workspace_base + 12845056;
  tensor695 = workspace_base + 25690112;
  tensor676 = workspace_base + 38535168;
  tensor751 = workspace_base + 51380224;

  // Data & mask generation
  generate_data(tensor635, 12845056);
  generate_data(tensor691, 12845056);
  generate_data(tensor695, 12845056);
  generate_data(tensor676, 12845056);
  generate_data(tensor751, 12845056);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
