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
float *tensor579, *tensor580, *tensor581, *tensor583, *tensor585, *tensor582;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // Conv2d Size: 524288
  apply_grad<<<64, 0>>>(tensor581, tensor582, 524288);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("272 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 158334976);
  
  // Weight addr assignment
  tensor579 = workspace_base + 0;
  tensor580 = workspace_base + 12845056;
  tensor581 = workspace_base + 19267584;
  tensor583 = workspace_base + 19791872;
  tensor585 = workspace_base + 32636928;
  tensor582 = workspace_base + 39059456;

  // Data & mask generation
  generate_data(tensor579, 12845056);
  generate_data(tensor580, 6422528);
  generate_data(tensor581, 524288);
  generate_data(tensor583, 12845056);
  generate_data(tensor585, 6422528);
  generate_data(tensor582, 524288);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
