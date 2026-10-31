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
float *tensor602, *tensor616, *tensor617, *tensor619, *tensor621, *tensor618;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][512][14][14], Out:[64][1024][14][14], Weight:[1024][512][1][1]
  forwardPass_conv2d_l86<<<64, 0>>>((float (*)[512][14][14]) tensor602, (float (*)[1024][14][14]) tensor616, (float (*)[512][1][1]) tensor617);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("087 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 158334976);
  
  // Weight addr assignment
  tensor602 = workspace_base + 0;
  tensor616 = workspace_base + 6422528;
  tensor617 = workspace_base + 19267584;
  tensor619 = workspace_base + 19791872;
  tensor621 = workspace_base + 26214400;
  tensor618 = workspace_base + 39059456;

  // Data & mask generation
  generate_data(tensor602, 6422528);
  generate_data(tensor616, 12845056);
  generate_data(tensor617, 524288);
  generate_data(tensor619, 6422528);
  generate_data(tensor621, 12845056);
  generate_data(tensor618, 524288);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
