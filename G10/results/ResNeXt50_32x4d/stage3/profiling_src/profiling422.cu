#include "../include/declaration6.cuh"
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
float *tensor20, *tensor77, *tensor78, *tensor80, *tensor82, *tensor79;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][64][56][56], D_out:[64][256][56][56], D_weight:[256][64][1][1]
  backwardPass_conv2d_Weight_l11<<<64, 0>>>((float (*)[64][56][56]) tensor20, (float (*)[256][56][56]) tensor82, (float (*)[64][1][1]) tensor79);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("422 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 513933312);
  
  // Weight addr assignment
  tensor20 = workspace_base + 0;
  tensor77 = workspace_base + 12845056;
  tensor78 = workspace_base + 64225280;
  tensor80 = workspace_base + 64241664;
  tensor82 = workspace_base + 77086720;
  tensor79 = workspace_base + 128466944;

  // Data & mask generation
  generate_data(tensor20, 12845056);
  generate_data(tensor77, 51380224);
  generate_data(tensor78, 16384);
  generate_data(tensor80, 12845056);
  generate_data(tensor82, 51380224);
  generate_data(tensor79, 16384);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
