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
float *tensor5, *tensor20, *tensor21, *tensor25, *tensor80;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][64][112][112], Out:[64][64][56][56]
  forwardPass_MaxPool2d_l3<<<64, 0>>>((float (*)[64][112][112]) tensor5, (float (*)[64][56][56]) tensor20);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("004 %20.10f ms\n", elapsedTime);
}

00// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 565182464);
  
  // Weight addr assignment
  tensor5 = workspace_base + 0;
  tensor20 = workspace_base + 51380224;
  tensor21 = workspace_base + 64225280;
  tensor25 = workspace_base + 115605504;
  tensor80 = workspace_base + 128450560;

  // Data & mask generation
  generate_data(tensor5, 51380224);
  generate_data(tensor20, 12845056);
  generate_data(tensor21, 51380224);
  generate_data(tensor25, 12845056);
  generate_data(tensor80, 12845056);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
