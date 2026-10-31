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
float *tensor0, *tensor1, *tensor2, *tensor4, *tensor6, *tensor3;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][3][224][224], D_out:[64][64][112][112], D_weight:[64][3][7][7]
  backwardPass_conv2d_Weight_l0<<<64, 0>>>((float (*)[3][224][224]) tensor0, (float (*)[64][112][112]) tensor6, (float (*)[3][7][7]) tensor3);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("446 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 488194048);
  
  // Weight addr assignment
  tensor0 = workspace_base + 0;
  tensor1 = workspace_base + 9633792;
  tensor2 = workspace_base + 61014016;
  tensor4 = workspace_base + 61024256;
  tensor6 = workspace_base + 70658048;
  tensor3 = workspace_base + 122038272;

  // Data & mask generation
  generate_data(tensor0, 9633792);
  generate_data(tensor1, 51380224);
  generate_data(tensor2, 10240);
  generate_data(tensor4, 9633792);
  generate_data(tensor6, 51380224);
  generate_data(tensor3, 10240);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
