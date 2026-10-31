#include "../include/declaration2.cuh"
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
float *tensor974, *tensor974, *tensor988, *tensor991;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_in:[64][2048][7][7], In:[64][2048][7][7], D_output:[64][2048][7][7]
  backwardPass_ReLU_l138<<<64, 0>>>((float (*)[2048][7][7]) tensor988, (float (*)[2048][7][7]) tensor974, (float (*)[2048][7][7]) tensor991);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("150 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 102760448);
  
  // Weight addr assignment
  tensor974 = workspace_base + 0;
  tensor974 = workspace_base + 6422528;
  tensor988 = workspace_base + 12845056;
  tensor991 = workspace_base + 19267584;

  // Data & mask generation
  generate_data(tensor974, 6422528);
  generate_data(tensor974, 6422528);
  generate_data(tensor988, 6422528);
  generate_data(tensor991, 6422528);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
