#include "../include/declaration3.cuh"
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
float *tensor770, *tensor784, *tensor785, *tensor787, *tensor789, *tensor786;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][512][14][14], D_out:[64][1024][14][14], D_weight:[1024][512][1][1]
  backwardPass_conv2d_Weight_l110<<<64, 0>>>((float (*)[512][14][14]) tensor770, (float (*)[1024][14][14]) tensor789, (float (*)[512][1][1]) tensor786);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("209 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 158334976);
  
  // Weight addr assignment
  tensor770 = workspace_base + 0;
  tensor784 = workspace_base + 6422528;
  tensor785 = workspace_base + 19267584;
  tensor787 = workspace_base + 19791872;
  tensor789 = workspace_base + 26214400;
  tensor786 = workspace_base + 39059456;

  // Data & mask generation
  generate_data(tensor770, 6422528);
  generate_data(tensor784, 12845056);
  generate_data(tensor785, 524288);
  generate_data(tensor787, 6422528);
  generate_data(tensor789, 12845056);
  generate_data(tensor786, 524288);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
