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
float *tensor546, *tensor560, *tensor561, *tensor563, *tensor565, *tensor562;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // Conv2d Size: 524288
  apply_grad<<<64, 0>>>(tensor561, tensor562, 524288);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("279 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 158334976);
  
  // Weight addr assignment
  tensor546 = workspace_base + 0;
  tensor560 = workspace_base + 6422528;
  tensor561 = workspace_base + 19267584;
  tensor563 = workspace_base + 19791872;
  tensor565 = workspace_base + 26214400;
  tensor562 = workspace_base + 39059456;

  // Data & mask generation
  generate_data(tensor546, 6422528);
  generate_data(tensor560, 12845056);
  generate_data(tensor561, 524288);
  generate_data(tensor563, 6422528);
  generate_data(tensor565, 12845056);
  generate_data(tensor562, 524288);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
