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
float *tensor156, *tensor170, *tensor171, *tensor173, *tensor175, *tensor172;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][128][56][56], D_out:[64][128][56][56], D_weight:[128][128][3][3]
  backwardPass_conv2d_Weight_l24<<<64, 0>>>((float (*)[128][56][56]) tensor156, (float (*)[128][56][56]) tensor175, (float (*)[128][3][3]) tensor172);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("394 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 411082752);
  
  // Weight addr assignment
  tensor156 = workspace_base + 0;
  tensor170 = workspace_base + 25690112;
  tensor171 = workspace_base + 51380224;
  tensor173 = workspace_base + 51385344;
  tensor175 = workspace_base + 77075456;
  tensor172 = workspace_base + 102765568;

  // Data & mask generation
  generate_data(tensor156, 25690112);
  generate_data(tensor170, 25690112);
  generate_data(tensor171, 5120);
  generate_data(tensor173, 25690112);
  generate_data(tensor175, 25690112);
  generate_data(tensor172, 5120);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
