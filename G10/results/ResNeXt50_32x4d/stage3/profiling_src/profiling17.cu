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
float *tensor100, *tensor114, *tensor115, *tensor117, *tensor119, *tensor116;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][128][56][56], Out:[64][128][56][56], Weight:[128][128][3][3]
  forwardPass_conv2d_l16<<<64, 0>>>((float (*)[128][56][56]) tensor100, (float (*)[128][56][56]) tensor114, (float (*)[128][3][3]) tensor115);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("017 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 411082752);
  
  // Weight addr assignment
  tensor100 = workspace_base + 0;
  tensor114 = workspace_base + 25690112;
  tensor115 = workspace_base + 51380224;
  tensor117 = workspace_base + 51385344;
  tensor119 = workspace_base + 77075456;
  tensor116 = workspace_base + 102765568;

  // Data & mask generation
  generate_data(tensor100, 25690112);
  generate_data(tensor114, 25690112);
  generate_data(tensor115, 5120);
  generate_data(tensor117, 25690112);
  generate_data(tensor119, 25690112);
  generate_data(tensor116, 5120);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
