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
float *tensor468, *tensor472, *tensor473, *tensor489, *tensor474, *tensor475, *tensor484, *tensor485;
float *tensor476, *tensor477, *tensor478, *tensor479, *tensor480, *tensor481, *tensor482, *tensor483;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // BatchNorm Size: 1024
  apply_grad<<<64, 0>>>(tensor474, tensor475, 1024);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("303 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 154181632);
  
  // Weight addr assignment
  tensor468 = workspace_base + 0;
  tensor472 = workspace_base + 6422528;
  tensor473 = workspace_base + 12845056;
  tensor489 = workspace_base + 19267584;
  tensor474 = workspace_base + 25690112;
  tensor475 = workspace_base + 25691136;
  tensor484 = workspace_base + 25692160;
  tensor485 = workspace_base + 25693184;
  tensor476 = workspace_base + 25694208;
  tensor477 = workspace_base + 25695232;
  tensor478 = workspace_base + 25696256;
  tensor479 = workspace_base + 32118784;
  tensor480 = workspace_base + 32119808;
  tensor481 = workspace_base + 32120832;
  tensor482 = workspace_base + 32121856;
  tensor483 = workspace_base + 38544384;

  // Data & mask generation
  generate_data(tensor468, 6422528);
  generate_data(tensor472, 6422528);
  generate_data(tensor473, 6422528);
  generate_data(tensor489, 6422528);
  generate_data(tensor474, 1024);
  generate_data(tensor475, 1024);
  generate_data(tensor484, 1024);
  generate_data(tensor485, 1024);
  generate_data(tensor476, 1024);
  generate_data(tensor477, 1024);
  generate_data(tensor478, 6422528);
  generate_data(tensor479, 1024);
  generate_data(tensor480, 1024);
  generate_data(tensor481, 1024);
  generate_data(tensor482, 6422528);
  generate_data(tensor483, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
