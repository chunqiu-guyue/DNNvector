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
float *tensor342, *tensor356, *tensor357, *tensor359, *tensor361, *tensor358;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_in:[64][256][28][28], D_out:[64][256][28][28], Weight:[256][256][3][3]
  backwardPass_conv2d_Input_l50<<<64, 0>>>((float (*)[256][28][28]) tensor359, (float (*)[256][28][28]) tensor361, (float (*)[256][3][3]) tensor357);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("339 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 205668352);
  
  // Weight addr assignment
  tensor342 = workspace_base + 0;
  tensor356 = workspace_base + 12845056;
  tensor357 = workspace_base + 25690112;
  tensor359 = workspace_base + 25708544;
  tensor361 = workspace_base + 38553600;
  tensor358 = workspace_base + 51398656;

  // Data & mask generation
  generate_data(tensor342, 12845056);
  generate_data(tensor356, 12845056);
  generate_data(tensor357, 18432);
  generate_data(tensor359, 12845056);
  generate_data(tensor361, 12845056);
  generate_data(tensor358, 18432);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
