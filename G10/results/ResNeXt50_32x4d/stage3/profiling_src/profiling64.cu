#include "../include/declaration1.cuh"
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
float *tensor393, *tensor449, *tensor453, *tensor434, *tensor508;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // NumInputs:2, Size:25690112
  forwardPass_Add_l63<<<64, 0>>>(tensor393, tensor434, tensor449, 25690112);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("064 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 513802240);
  
  // Weight addr assignment
  tensor393 = workspace_base + 0;
  tensor449 = workspace_base + 25690112;
  tensor453 = workspace_base + 51380224;
  tensor434 = workspace_base + 77070336;
  tensor508 = workspace_base + 102760448;

  // Data & mask generation
  generate_data(tensor393, 25690112);
  generate_data(tensor449, 25690112);
  generate_data(tensor453, 25690112);
  generate_data(tensor434, 25690112);
  generate_data(tensor508, 25690112);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
