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
float *tensor449, *tensor450, *tensor451, *tensor453, *tensor455, *tensor452;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][512][28][28], Out:[64][512][28][28], Weight:[512][512][1][1]
  forwardPass_conv2d_l64<<<64, 0>>>((float (*)[512][28][28]) tensor449, (float (*)[512][28][28]) tensor450, (float (*)[512][1][1]) tensor451);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("065 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 413138944);
  
  // Weight addr assignment
  tensor449 = workspace_base + 0;
  tensor450 = workspace_base + 25690112;
  tensor451 = workspace_base + 51380224;
  tensor453 = workspace_base + 51642368;
  tensor455 = workspace_base + 77332480;
  tensor452 = workspace_base + 103022592;

  // Data & mask generation
  generate_data(tensor449, 25690112);
  generate_data(tensor450, 25690112);
  generate_data(tensor451, 262144);
  generate_data(tensor453, 25690112);
  generate_data(tensor455, 25690112);
  generate_data(tensor452, 262144);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
