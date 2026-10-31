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
float *tensor286, *tensor300, *tensor301, *tensor303, *tensor305, *tensor302;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // Conv2d Size: 18432
  apply_grad<<<64, 0>>>(tensor301, tensor302, 18432);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("357 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 205668352);
  
  // Weight addr assignment
  tensor286 = workspace_base + 0;
  tensor300 = workspace_base + 12845056;
  tensor301 = workspace_base + 25690112;
  tensor303 = workspace_base + 25708544;
  tensor305 = workspace_base + 38553600;
  tensor302 = workspace_base + 51398656;

  // Data & mask generation
  generate_data(tensor286, 12845056);
  generate_data(tensor300, 12845056);
  generate_data(tensor301, 18432);
  generate_data(tensor303, 12845056);
  generate_data(tensor305, 12845056);
  generate_data(tensor302, 18432);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
