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
float *tensor990, *tensor993, *tensor992, *tensor994, *tensor998, *tensor995, *tensor996, *tensor997;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // MakeError Size: 258048
  makeError<<<64, 0>>>(tensor998, tensor993, 0, 64512);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("143 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 17956864);
  
  // Weight addr assignment
  tensor990 = workspace_base + 0;
  tensor993 = workspace_base + 131072;
  tensor992 = workspace_base + 195584;
  tensor994 = workspace_base + 2243584;
  tensor998 = workspace_base + 2374656;
  tensor995 = workspace_base + 2439168;
  tensor996 = workspace_base + 4487168;
  tensor997 = workspace_base + 4488192;

  // Data & mask generation
  generate_data(tensor990, 131072);
  generate_data(tensor993, 64512);
  generate_data(tensor992, 2048000);
  generate_data(tensor994, 131072);
  generate_data(tensor998, 64512);
  generate_data(tensor995, 2048000);
  generate_data(tensor996, 1024);
  generate_data(tensor997, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
