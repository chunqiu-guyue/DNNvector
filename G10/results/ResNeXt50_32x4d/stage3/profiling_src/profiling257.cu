#include "../include/declaration4.cuh"
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
float *tensor620, *tensor620, *tensor634, *tensor639;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_in:[64][1024][14][14], In:[64][1024][14][14], D_output:[64][1024][14][14]
  backwardPass_ReLU_l88<<<64, 0>>>((float (*)[1024][14][14]) tensor634, (float (*)[1024][14][14]) tensor620, (float (*)[1024][14][14]) tensor639);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("257 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 205520896);
  
  // Weight addr assignment
  tensor620 = workspace_base + 0;
  tensor620 = workspace_base + 12845056;
  tensor634 = workspace_base + 25690112;
  tensor639 = workspace_base + 38535168;

  // Data & mask generation
  generate_data(tensor620, 12845056);
  generate_data(tensor620, 12845056);
  generate_data(tensor634, 12845056);
  generate_data(tensor639, 12845056);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
