#include "../include/declaration6.cuh"
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
float *tensor20, *tensor22, *tensor23, *tensor25, *tensor27, *tensor24;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_in:[64][64][56][56], D_out:[64][128][56][56], Weight:[128][64][1][1]
  backwardPass_conv2d_Input_l4<<<64, 0>>>((float (*)[64][56][56]) tensor25, (float (*)[128][56][56]) tensor27, (float (*)[64][1][1]) tensor23);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("439 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 308346880);
  
  // Weight addr assignment
  tensor20 = workspace_base + 0;
  tensor22 = workspace_base + 12845056;
  tensor23 = workspace_base + 38535168;
  tensor25 = workspace_base + 38543360;
  tensor27 = workspace_base + 51388416;
  tensor24 = workspace_base + 77078528;

  // Data & mask generation
  generate_data(tensor20, 12845056);
  generate_data(tensor22, 25690112);
  generate_data(tensor23, 8192);
  generate_data(tensor25, 12845056);
  generate_data(tensor27, 25690112);
  generate_data(tensor24, 8192);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
