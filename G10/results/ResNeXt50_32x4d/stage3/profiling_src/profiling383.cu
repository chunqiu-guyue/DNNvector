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
float *tensor207, *tensor208, *tensor209, *tensor211, *tensor213, *tensor210;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_in:[64][256][56][56], D_out:[64][256][56][56], Weight:[256][256][1][1]
  backwardPass_conv2d_Input_l30<<<64, 0>>>((float (*)[256][56][56]) tensor211, (float (*)[256][56][56]) tensor213, (float (*)[256][1][1]) tensor209);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("383 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 822607872);
  
  // Weight addr assignment
  tensor207 = workspace_base + 0;
  tensor208 = workspace_base + 51380224;
  tensor209 = workspace_base + 102760448;
  tensor211 = workspace_base + 102825984;
  tensor213 = workspace_base + 154206208;
  tensor210 = workspace_base + 205586432;

  // Data & mask generation
  generate_data(tensor207, 51380224);
  generate_data(tensor208, 51380224);
  generate_data(tensor209, 65536);
  generate_data(tensor211, 51380224);
  generate_data(tensor213, 51380224);
  generate_data(tensor210, 65536);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
