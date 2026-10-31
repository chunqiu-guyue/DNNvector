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
float *tensor449, *tensor505, *tensor506, *tensor508, *tensor510, *tensor507;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // Conv2d Size: 524288
  apply_grad<<<64, 0>>>(tensor506, tensor507, 524288);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("295 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 312475648);
  
  // Weight addr assignment
  tensor449 = workspace_base + 0;
  tensor505 = workspace_base + 25690112;
  tensor506 = workspace_base + 38535168;
  tensor508 = workspace_base + 39059456;
  tensor510 = workspace_base + 64749568;
  tensor507 = workspace_base + 77594624;

  // Data & mask generation
  generate_data(tensor449, 25690112);
  generate_data(tensor505, 12845056);
  generate_data(tensor506, 524288);
  generate_data(tensor508, 25690112);
  generate_data(tensor510, 12845056);
  generate_data(tensor507, 524288);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
