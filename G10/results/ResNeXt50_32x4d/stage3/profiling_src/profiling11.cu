#include "../include/declaration0.cuh"
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
float *tensor62, *tensor62, *tensor76, *tensor99;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][256][56][56], Out:[64][256][56][56]
  forwardPass_ReLU_l10<<<64, 0>>>((float (*)[256][56][56]) tensor62, (float (*)[256][56][56]) tensor62);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("011 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 822083584);
  
  // Weight addr assignment
  tensor62 = workspace_base + 0;
  tensor62 = workspace_base + 51380224;
  tensor76 = workspace_base + 102760448;
  tensor99 = workspace_base + 154140672;

  // Data & mask generation
  generate_data(tensor62, 51380224);
  generate_data(tensor62, 51380224);
  generate_data(tensor76, 51380224);
  generate_data(tensor99, 51380224);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
