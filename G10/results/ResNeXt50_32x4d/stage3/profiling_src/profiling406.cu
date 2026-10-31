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
float *tensor118, *tensor132, *tensor133, *tensor135, *tensor137, *tensor134;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][128][56][56], D_out:[64][256][56][56], D_weight:[256][128][1][1]
  backwardPass_conv2d_Weight_l18<<<64, 0>>>((float (*)[128][56][56]) tensor118, (float (*)[256][56][56]) tensor137, (float (*)[128][1][1]) tensor134);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("406 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 616824832);
  
  // Weight addr assignment
  tensor118 = workspace_base + 0;
  tensor132 = workspace_base + 25690112;
  tensor133 = workspace_base + 77070336;
  tensor135 = workspace_base + 77103104;
  tensor137 = workspace_base + 102793216;
  tensor134 = workspace_base + 154173440;

  // Data & mask generation
  generate_data(tensor118, 25690112);
  generate_data(tensor132, 51380224);
  generate_data(tensor133, 32768);
  generate_data(tensor135, 25690112);
  generate_data(tensor137, 51380224);
  generate_data(tensor134, 32768);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
