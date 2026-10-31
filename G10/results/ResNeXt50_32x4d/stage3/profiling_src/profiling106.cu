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
float *tensor691, *tensor747, *tensor751, *tensor732, *tensor807;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // NumInputs:2, Size:12845056
  forwardPass_Add_l105<<<64, 0>>>(tensor691, tensor732, tensor747, 12845056);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("106 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 256901120);
  
  // Weight addr assignment
  tensor691 = workspace_base + 0;
  tensor747 = workspace_base + 12845056;
  tensor751 = workspace_base + 25690112;
  tensor732 = workspace_base + 38535168;
  tensor807 = workspace_base + 51380224;

  // Data & mask generation
  generate_data(tensor691, 12845056);
  generate_data(tensor747, 12845056);
  generate_data(tensor751, 12845056);
  generate_data(tensor732, 12845056);
  generate_data(tensor807, 12845056);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
