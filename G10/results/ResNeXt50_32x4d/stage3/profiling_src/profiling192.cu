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
float *tensor826, *tensor840, *tensor841, *tensor843, *tensor845, *tensor842;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][1024][7][7], D_out:[64][2048][7][7], D_weight:[2048][1024][1][1]
  backwardPass_conv2d_Weight_l118<<<64, 0>>>((float (*)[1024][7][7]) tensor826, (float (*)[2048][7][7]) tensor845, (float (*)[1024][1][1]) tensor842);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("192 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 93847552);
  
  // Weight addr assignment
  tensor826 = workspace_base + 0;
  tensor840 = workspace_base + 3211264;
  tensor841 = workspace_base + 9633792;
  tensor843 = workspace_base + 11730944;
  tensor845 = workspace_base + 14942208;
  tensor842 = workspace_base + 21364736;

  // Data & mask generation
  generate_data(tensor826, 3211264);
  generate_data(tensor840, 6422528);
  generate_data(tensor841, 2097152);
  generate_data(tensor843, 3211264);
  generate_data(tensor845, 6422528);
  generate_data(tensor842, 2097152);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
