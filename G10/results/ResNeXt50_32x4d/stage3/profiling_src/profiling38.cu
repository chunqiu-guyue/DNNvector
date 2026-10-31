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
float *tensor207, *tensor263, *tensor264, *tensor266, *tensor268, *tensor265;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][256][56][56], Out:[64][512][28][28], Weight:[512][256][1][1]
  forwardPass_conv2d_l37<<<64, 0>>>((float (*)[256][56][56]) tensor207, (float (*)[512][28][28]) tensor263, (float (*)[256][1][1]) tensor264);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("038 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 617611264);
  
  // Weight addr assignment
  tensor207 = workspace_base + 0;
  tensor263 = workspace_base + 51380224;
  tensor264 = workspace_base + 77070336;
  tensor266 = workspace_base + 77201408;
  tensor268 = workspace_base + 128581632;
  tensor265 = workspace_base + 154271744;

  // Data & mask generation
  generate_data(tensor207, 51380224);
  generate_data(tensor263, 25690112);
  generate_data(tensor264, 131072);
  generate_data(tensor266, 51380224);
  generate_data(tensor268, 25690112);
  generate_data(tensor265, 131072);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
