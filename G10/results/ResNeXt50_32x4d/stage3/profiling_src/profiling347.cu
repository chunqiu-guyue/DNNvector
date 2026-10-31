#include "../include/declaration5.cuh"
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
float *tensor322, *tensor322, *tensor336, *tensor341;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_in:[64][512][28][28], In:[64][512][28][28], D_output:[64][512][28][28]
  backwardPass_ReLU_l46<<<64, 0>>>((float (*)[512][28][28]) tensor336, (float (*)[512][28][28]) tensor322, (float (*)[512][28][28]) tensor341);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("347 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 411041792);
  
  // Weight addr assignment
  tensor322 = workspace_base + 0;
  tensor322 = workspace_base + 25690112;
  tensor336 = workspace_base + 51380224;
  tensor341 = workspace_base + 77070336;

  // Data & mask generation
  generate_data(tensor322, 25690112);
  generate_data(tensor322, 25690112);
  generate_data(tensor336, 25690112);
  generate_data(tensor341, 25690112);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
