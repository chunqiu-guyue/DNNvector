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
float *tensor374, *tensor378, *tensor379, *tensor392, *tensor380, *tensor381, *tensor390, *tensor391;
float *tensor382, *tensor383, *tensor384, *tensor385, *tensor386, *tensor387, *tensor388, *tensor389;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // BatchNorm Size: 1024
  apply_grad<<<64, 0>>>(tensor380, tensor381, 1024);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("332 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 616603648);
  
  // Weight addr assignment
  tensor374 = workspace_base + 0;
  tensor378 = workspace_base + 25690112;
  tensor379 = workspace_base + 51380224;
  tensor392 = workspace_base + 77070336;
  tensor380 = workspace_base + 102760448;
  tensor381 = workspace_base + 102761472;
  tensor390 = workspace_base + 102762496;
  tensor391 = workspace_base + 102763520;
  tensor382 = workspace_base + 102764544;
  tensor383 = workspace_base + 102765568;
  tensor384 = workspace_base + 102766592;
  tensor385 = workspace_base + 128456704;
  tensor386 = workspace_base + 128457728;
  tensor387 = workspace_base + 128458752;
  tensor388 = workspace_base + 128459776;
  tensor389 = workspace_base + 154149888;

  // Data & mask generation
  generate_data(tensor374, 25690112);
  generate_data(tensor378, 25690112);
  generate_data(tensor379, 25690112);
  generate_data(tensor392, 25690112);
  generate_data(tensor380, 1024);
  generate_data(tensor381, 1024);
  generate_data(tensor390, 1024);
  generate_data(tensor391, 1024);
  generate_data(tensor382, 1024);
  generate_data(tensor383, 1024);
  generate_data(tensor384, 25690112);
  generate_data(tensor385, 1024);
  generate_data(tensor386, 1024);
  generate_data(tensor387, 1024);
  generate_data(tensor388, 25690112);
  generate_data(tensor389, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
