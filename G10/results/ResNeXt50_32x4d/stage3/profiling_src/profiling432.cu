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
float *tensor40, *tensor44, *tensor45, *tensor61, *tensor46, *tensor47, *tensor56, *tensor57;
float *tensor48, *tensor49, *tensor50, *tensor51, *tensor52, *tensor53, *tensor54, *tensor55;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // BatchNorm Size: 1024
  apply_grad<<<64, 0>>>(tensor46, tensor47, 1024);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("432 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 616603648);
  
  // Weight addr assignment
  tensor40 = workspace_base + 0;
  tensor44 = workspace_base + 25690112;
  tensor45 = workspace_base + 51380224;
  tensor61 = workspace_base + 77070336;
  tensor46 = workspace_base + 102760448;
  tensor47 = workspace_base + 102761472;
  tensor56 = workspace_base + 102762496;
  tensor57 = workspace_base + 102763520;
  tensor48 = workspace_base + 102764544;
  tensor49 = workspace_base + 102765568;
  tensor50 = workspace_base + 102766592;
  tensor51 = workspace_base + 128456704;
  tensor52 = workspace_base + 128457728;
  tensor53 = workspace_base + 128458752;
  tensor54 = workspace_base + 128459776;
  tensor55 = workspace_base + 154149888;

  // Data & mask generation
  generate_data(tensor40, 25690112);
  generate_data(tensor44, 25690112);
  generate_data(tensor45, 25690112);
  generate_data(tensor61, 25690112);
  generate_data(tensor46, 1024);
  generate_data(tensor47, 1024);
  generate_data(tensor56, 1024);
  generate_data(tensor57, 1024);
  generate_data(tensor48, 1024);
  generate_data(tensor49, 1024);
  generate_data(tensor50, 25690112);
  generate_data(tensor51, 1024);
  generate_data(tensor52, 1024);
  generate_data(tensor53, 1024);
  generate_data(tensor54, 25690112);
  generate_data(tensor55, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
