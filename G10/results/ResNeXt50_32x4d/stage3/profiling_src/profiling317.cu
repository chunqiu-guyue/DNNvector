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
float *tensor416, *tensor430, *tensor431, *tensor433, *tensor435, *tensor432;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_in:[64][256][28][28], D_out:[64][512][28][28], Weight:[512][256][1][1]
  backwardPass_conv2d_Input_l60<<<64, 0>>>((float (*)[256][28][28]) tensor433, (float (*)[512][28][28]) tensor435, (float (*)[256][1][1]) tensor431);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("317 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 309329920);
  
  // Weight addr assignment
  tensor416 = workspace_base + 0;
  tensor430 = workspace_base + 12845056;
  tensor431 = workspace_base + 38535168;
  tensor433 = workspace_base + 38666240;
  tensor435 = workspace_base + 51511296;
  tensor432 = workspace_base + 77201408;

  // Data & mask generation
  generate_data(tensor416, 12845056);
  generate_data(tensor430, 25690112);
  generate_data(tensor431, 131072);
  generate_data(tensor433, 12845056);
  generate_data(tensor435, 25690112);
  generate_data(tensor432, 131072);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
