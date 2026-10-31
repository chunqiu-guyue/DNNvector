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
float *tensor360, *tensor374, *tensor375, *tensor377, *tensor379, *tensor376;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // Conv2d Size: 131072
  apply_grad<<<64, 0>>>(tensor375, tensor376, 131072);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("335 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 309329920);
  
  // Weight addr assignment
  tensor360 = workspace_base + 0;
  tensor374 = workspace_base + 12845056;
  tensor375 = workspace_base + 38535168;
  tensor377 = workspace_base + 38666240;
  tensor379 = workspace_base + 51511296;
  tensor376 = workspace_base + 77201408;

  // Data & mask generation
  generate_data(tensor360, 12845056);
  generate_data(tensor374, 25690112);
  generate_data(tensor375, 131072);
  generate_data(tensor377, 12845056);
  generate_data(tensor379, 25690112);
  generate_data(tensor376, 131072);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
