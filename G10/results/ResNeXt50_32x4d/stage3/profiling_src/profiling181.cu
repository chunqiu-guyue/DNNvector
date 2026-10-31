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
float *tensor877, *tensor878, *tensor879, *tensor881, *tensor883, *tensor880;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_in:[64][2048][7][7], D_out:[64][1024][7][7], Weight:[1024][2048][1][1]
  backwardPass_conv2d_Input_l124<<<64, 0>>>((float (*)[2048][7][7]) tensor881, (float (*)[1024][7][7]) tensor883, (float (*)[2048][1][1]) tensor879);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("181 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 93847552);
  
  // Weight addr assignment
  tensor877 = workspace_base + 0;
  tensor878 = workspace_base + 6422528;
  tensor879 = workspace_base + 9633792;
  tensor881 = workspace_base + 11730944;
  tensor883 = workspace_base + 18153472;
  tensor880 = workspace_base + 21364736;

  // Data & mask generation
  generate_data(tensor877, 6422528);
  generate_data(tensor878, 3211264);
  generate_data(tensor879, 2097152);
  generate_data(tensor881, 6422528);
  generate_data(tensor883, 3211264);
  generate_data(tensor880, 2097152);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
