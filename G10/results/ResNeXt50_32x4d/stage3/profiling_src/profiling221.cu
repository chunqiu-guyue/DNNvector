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
float *tensor747, *tensor748, *tensor749, *tensor751, *tensor753, *tensor750;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // Conv2d Size: 524288
  apply_grad<<<64, 0>>>(tensor749, tensor750, 524288);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("221 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 158334976);
  
  // Weight addr assignment
  tensor747 = workspace_base + 0;
  tensor748 = workspace_base + 12845056;
  tensor749 = workspace_base + 19267584;
  tensor751 = workspace_base + 19791872;
  tensor753 = workspace_base + 32636928;
  tensor750 = workspace_base + 39059456;

  // Data & mask generation
  generate_data(tensor747, 12845056);
  generate_data(tensor748, 6422528);
  generate_data(tensor749, 524288);
  generate_data(tensor751, 12845056);
  generate_data(tensor753, 6422528);
  generate_data(tensor750, 524288);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
