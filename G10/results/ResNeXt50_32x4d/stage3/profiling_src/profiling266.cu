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
float *tensor584, *tensor598, *tensor599, *tensor601, *tensor603, *tensor600;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_in:[64][512][14][14], D_out:[64][512][14][14], Weight:[512][512][3][3]
  backwardPass_conv2d_Input_l84<<<64, 0>>>((float (*)[512][14][14]) tensor601, (float (*)[512][14][14]) tensor603, (float (*)[512][3][3]) tensor599);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("266 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 103350272);
  
  // Weight addr assignment
  tensor584 = workspace_base + 0;
  tensor598 = workspace_base + 6422528;
  tensor599 = workspace_base + 12845056;
  tensor601 = workspace_base + 12918784;
  tensor603 = workspace_base + 19341312;
  tensor600 = workspace_base + 25763840;

  // Data & mask generation
  generate_data(tensor584, 6422528);
  generate_data(tensor598, 6422528);
  generate_data(tensor599, 73728);
  generate_data(tensor601, 6422528);
  generate_data(tensor603, 6422528);
  generate_data(tensor600, 73728);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
