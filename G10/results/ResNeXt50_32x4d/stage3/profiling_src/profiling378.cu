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
float *tensor212, *tensor226, *tensor227, *tensor229, *tensor231, *tensor228;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_in:[64][256][56][56], D_out:[64][256][28][28], Weight:[256][256][3][3]
  backwardPass_conv2d_Input_l32<<<64, 0>>>((float (*)[256][56][56]) tensor229, (float (*)[256][28][28]) tensor231, (float (*)[256][3][3]) tensor227);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("378 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 513949696);
  
  // Weight addr assignment
  tensor212 = workspace_base + 0;
  tensor226 = workspace_base + 51380224;
  tensor227 = workspace_base + 64225280;
  tensor229 = workspace_base + 64243712;
  tensor231 = workspace_base + 115623936;
  tensor228 = workspace_base + 128468992;

  // Data & mask generation
  generate_data(tensor212, 51380224);
  generate_data(tensor226, 12845056);
  generate_data(tensor227, 18432);
  generate_data(tensor229, 51380224);
  generate_data(tensor231, 12845056);
  generate_data(tensor228, 18432);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
