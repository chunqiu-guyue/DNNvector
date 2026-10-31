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
float *tensor398, *tensor412, *tensor413, *tensor415, *tensor417, *tensor414;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][256][28][28], D_out:[64][256][28][28], D_weight:[256][256][3][3]
  backwardPass_conv2d_Weight_l58<<<64, 0>>>((float (*)[256][28][28]) tensor398, (float (*)[256][28][28]) tensor417, (float (*)[256][3][3]) tensor414);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("321 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 205668352);
  
  // Weight addr assignment
  tensor398 = workspace_base + 0;
  tensor412 = workspace_base + 12845056;
  tensor413 = workspace_base + 25690112;
  tensor415 = workspace_base + 25708544;
  tensor417 = workspace_base + 38553600;
  tensor414 = workspace_base + 51398656;

  // Data & mask generation
  generate_data(tensor398, 12845056);
  generate_data(tensor412, 12845056);
  generate_data(tensor413, 18432);
  generate_data(tensor415, 12845056);
  generate_data(tensor417, 12845056);
  generate_data(tensor414, 18432);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
