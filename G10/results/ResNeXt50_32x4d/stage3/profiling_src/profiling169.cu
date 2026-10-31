#include "../include/cudadnnUtil.cuh"
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
float *tensor914, *tensor918, *tensor919, *tensor932, *tensor920, *tensor921, *tensor930, *tensor931;
float *tensor922, *tensor923, *tensor924, *tensor925, *tensor926, *tensor927, *tensor928, *tensor929;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // BatchNorm Size: 4096
  apply_grad<<<64, 0>>>(tensor920, tensor921, 4096);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("169 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 154238976);
  
  // Weight addr assignment
  tensor914 = workspace_base + 0;
  tensor918 = workspace_base + 6422528;
  tensor919 = workspace_base + 12845056;
  tensor932 = workspace_base + 19267584;
  tensor920 = workspace_base + 25690112;
  tensor921 = workspace_base + 25694208;
  tensor930 = workspace_base + 25698304;
  tensor931 = workspace_base + 25700352;
  tensor922 = workspace_base + 25702400;
  tensor923 = workspace_base + 25704448;
  tensor924 = workspace_base + 25706496;
  tensor925 = workspace_base + 32129024;
  tensor926 = workspace_base + 32131072;
  tensor927 = workspace_base + 32133120;
  tensor928 = workspace_base + 32135168;
  tensor929 = workspace_base + 38557696;

  // Data & mask generation
  generate_data(tensor914, 6422528);
  generate_data(tensor918, 6422528);
  generate_data(tensor919, 6422528);
  generate_data(tensor932, 6422528);
  generate_data(tensor920, 4096);
  generate_data(tensor921, 4096);
  generate_data(tensor930, 2048);
  generate_data(tensor931, 2048);
  generate_data(tensor922, 2048);
  generate_data(tensor923, 2048);
  generate_data(tensor924, 6422528);
  generate_data(tensor925, 2048);
  generate_data(tensor926, 2048);
  generate_data(tensor927, 2048);
  generate_data(tensor928, 6422528);
  generate_data(tensor929, 2048);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
