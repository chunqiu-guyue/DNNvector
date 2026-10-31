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
float *tensor304, *tensor318, *tensor319, *tensor321, *tensor323, *tensor320;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][256][28][28], D_out:[64][512][28][28], D_weight:[512][256][1][1]
  backwardPass_conv2d_Weight_l44<<<64, 0>>>((float (*)[256][28][28]) tensor304, (float (*)[512][28][28]) tensor323, (float (*)[256][1][1]) tensor320);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("350 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 309329920);
  
  // Weight addr assignment
  tensor304 = workspace_base + 0;
  tensor318 = workspace_base + 12845056;
  tensor319 = workspace_base + 38535168;
  tensor321 = workspace_base + 38666240;
  tensor323 = workspace_base + 51511296;
  tensor320 = workspace_base + 77201408;

  // Data & mask generation
  generate_data(tensor304, 12845056);
  generate_data(tensor318, 25690112);
  generate_data(tensor319, 131072);
  generate_data(tensor321, 12845056);
  generate_data(tensor323, 25690112);
  generate_data(tensor320, 131072);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
