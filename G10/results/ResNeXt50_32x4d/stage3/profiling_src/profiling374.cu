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
float *tensor230, *tensor244, *tensor245, *tensor247, *tensor249, *tensor246;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // Conv2d Size: 131072
  apply_grad<<<64, 0>>>(tensor245, tensor246, 131072);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("374 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 309329920);
  
  // Weight addr assignment
  tensor230 = workspace_base + 0;
  tensor244 = workspace_base + 12845056;
  tensor245 = workspace_base + 38535168;
  tensor247 = workspace_base + 38666240;
  tensor249 = workspace_base + 51511296;
  tensor246 = workspace_base + 77201408;

  // Data & mask generation
  generate_data(tensor230, 12845056);
  generate_data(tensor244, 25690112);
  generate_data(tensor245, 131072);
  generate_data(tensor247, 12845056);
  generate_data(tensor249, 25690112);
  generate_data(tensor246, 131072);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
