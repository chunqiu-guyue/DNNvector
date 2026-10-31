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
float *tensor151, *tensor152, *tensor153, *tensor155, *tensor157, *tensor154;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // Conv2d Size: 32768
  apply_grad<<<64, 0>>>(tensor153, tensor154, 32768);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("401 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 616824832);
  
  // Weight addr assignment
  tensor151 = workspace_base + 0;
  tensor152 = workspace_base + 51380224;
  tensor153 = workspace_base + 77070336;
  tensor155 = workspace_base + 77103104;
  tensor157 = workspace_base + 128483328;
  tensor154 = workspace_base + 154173440;

  // Data & mask generation
  generate_data(tensor151, 51380224);
  generate_data(tensor152, 25690112);
  generate_data(tensor153, 32768);
  generate_data(tensor155, 51380224);
  generate_data(tensor157, 25690112);
  generate_data(tensor154, 32768);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
