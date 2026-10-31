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
float *tensor933, *tensor934, *tensor935, *tensor937, *tensor939, *tensor936;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // Conv2d Size: 2097152
  apply_grad<<<64, 0>>>(tensor935, tensor936, 2097152);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("165 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 93847552);
  
  // Weight addr assignment
  tensor933 = workspace_base + 0;
  tensor934 = workspace_base + 6422528;
  tensor935 = workspace_base + 9633792;
  tensor937 = workspace_base + 11730944;
  tensor939 = workspace_base + 18153472;
  tensor936 = workspace_base + 21364736;

  // Data & mask generation
  generate_data(tensor933, 6422528);
  generate_data(tensor934, 3211264);
  generate_data(tensor935, 2097152);
  generate_data(tensor937, 6422528);
  generate_data(tensor939, 3211264);
  generate_data(tensor936, 2097152);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
