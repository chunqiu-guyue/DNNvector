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
float *tensor640, *tensor654, *tensor655, *tensor657, *tensor659, *tensor656;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][512][14][14], Out:[64][512][14][14], Weight:[512][512][3][3]
  forwardPass_conv2d_l92<<<64, 0>>>((float (*)[512][14][14]) tensor640, (float (*)[512][14][14]) tensor654, (float (*)[512][3][3]) tensor655);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("093 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 103350272);
  
  // Weight addr assignment
  tensor640 = workspace_base + 0;
  tensor654 = workspace_base + 6422528;
  tensor655 = workspace_base + 12845056;
  tensor657 = workspace_base + 12918784;
  tensor659 = workspace_base + 19341312;
  tensor656 = workspace_base + 25763840;

  // Data & mask generation
  generate_data(tensor640, 6422528);
  generate_data(tensor654, 6422528);
  generate_data(tensor655, 73728);
  generate_data(tensor657, 6422528);
  generate_data(tensor659, 6422528);
  generate_data(tensor656, 73728);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
