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
float *tensor714, *tensor728, *tensor729, *tensor731, *tensor733, *tensor730;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // Conv2d Size: 524288
  apply_grad<<<64, 0>>>(tensor729, tensor730, 524288);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("228 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 158334976);
  
  // Weight addr assignment
  tensor714 = workspace_base + 0;
  tensor728 = workspace_base + 6422528;
  tensor729 = workspace_base + 19267584;
  tensor731 = workspace_base + 19791872;
  tensor733 = workspace_base + 26214400;
  tensor730 = workspace_base + 39059456;

  // Data & mask generation
  generate_data(tensor714, 6422528);
  generate_data(tensor728, 12845056);
  generate_data(tensor729, 524288);
  generate_data(tensor731, 6422528);
  generate_data(tensor733, 12845056);
  generate_data(tensor730, 524288);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
