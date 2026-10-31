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
float *tensor263, *tensor267, *tensor268, *tensor285, *tensor269, *tensor270, *tensor279, *tensor280;
float *tensor271, *tensor272, *tensor273, *tensor274, *tensor275, *tensor276, *tensor277, *tensor278;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // BatchNorm Size: 1024
  apply_grad<<<64, 0>>>(tensor269, tensor270, 1024);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("365 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 616603648);
  
  // Weight addr assignment
  tensor263 = workspace_base + 0;
  tensor267 = workspace_base + 25690112;
  tensor268 = workspace_base + 51380224;
  tensor285 = workspace_base + 77070336;
  tensor269 = workspace_base + 102760448;
  tensor270 = workspace_base + 102761472;
  tensor279 = workspace_base + 102762496;
  tensor280 = workspace_base + 102763520;
  tensor271 = workspace_base + 102764544;
  tensor272 = workspace_base + 102765568;
  tensor273 = workspace_base + 102766592;
  tensor274 = workspace_base + 128456704;
  tensor275 = workspace_base + 128457728;
  tensor276 = workspace_base + 128458752;
  tensor277 = workspace_base + 128459776;
  tensor278 = workspace_base + 154149888;

  // Data & mask generation
  generate_data(tensor263, 25690112);
  generate_data(tensor267, 25690112);
  generate_data(tensor268, 25690112);
  generate_data(tensor285, 25690112);
  generate_data(tensor269, 1024);
  generate_data(tensor270, 1024);
  generate_data(tensor279, 1024);
  generate_data(tensor280, 1024);
  generate_data(tensor271, 1024);
  generate_data(tensor272, 1024);
  generate_data(tensor273, 25690112);
  generate_data(tensor274, 1024);
  generate_data(tensor275, 1024);
  generate_data(tensor276, 1024);
  generate_data(tensor277, 25690112);
  generate_data(tensor278, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
