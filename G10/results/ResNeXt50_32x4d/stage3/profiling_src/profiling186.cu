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
float *tensor803, *tensor859, *tensor860, *tensor862, *tensor864, *tensor861;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][1024][14][14], D_out:[64][2048][7][7], D_weight:[2048][1024][1][1]
  backwardPass_conv2d_Weight_l121<<<64, 0>>>((float (*)[1024][14][14]) tensor803, (float (*)[2048][7][7]) tensor864, (float (*)[1024][1][1]) tensor861);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("186 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 170917888);
  
  // Weight addr assignment
  tensor803 = workspace_base + 0;
  tensor859 = workspace_base + 12845056;
  tensor860 = workspace_base + 19267584;
  tensor862 = workspace_base + 21364736;
  tensor864 = workspace_base + 34209792;
  tensor861 = workspace_base + 40632320;

  // Data & mask generation
  generate_data(tensor803, 12845056);
  generate_data(tensor859, 6422528);
  generate_data(tensor860, 2097152);
  generate_data(tensor862, 12845056);
  generate_data(tensor864, 6422528);
  generate_data(tensor861, 2097152);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
