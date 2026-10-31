#include "../include/declaration2.cuh"
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
float *tensor989, *tensor990, *tensor991, *tensor994;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][2048][7][7], Out:[64][2048][1][1]
  forwardPass_avgAdaptivePool2d_l140<<<64, 0>>>((float (*)[2048][7][7]) tensor989, (float (*)[2048][1][1]) tensor990);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("141 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 52428800);
  
  // Weight addr assignment
  tensor989 = workspace_base + 0;
  tensor990 = workspace_base + 6422528;
  tensor991 = workspace_base + 6553600;
  tensor994 = workspace_base + 12976128;

  // Data & mask generation
  generate_data(tensor989, 6422528);
  generate_data(tensor990, 131072);
  generate_data(tensor991, 6422528);
  generate_data(tensor994, 131072);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
