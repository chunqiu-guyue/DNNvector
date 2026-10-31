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
float *tensor934, *tensor938, *tensor939, *tensor955, *tensor940, *tensor941, *tensor950, *tensor951;
float *tensor942, *tensor943, *tensor944, *tensor945, *tensor946, *tensor947, *tensor948, *tensor949;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // BatchNorm Size: 2048
  apply_grad<<<64, 0>>>(tensor940, tensor941, 2048);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("162 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 77119488);
  
  // Weight addr assignment
  tensor934 = workspace_base + 0;
  tensor938 = workspace_base + 3211264;
  tensor939 = workspace_base + 6422528;
  tensor955 = workspace_base + 9633792;
  tensor940 = workspace_base + 12845056;
  tensor941 = workspace_base + 12847104;
  tensor950 = workspace_base + 12849152;
  tensor951 = workspace_base + 12850176;
  tensor942 = workspace_base + 12851200;
  tensor943 = workspace_base + 12852224;
  tensor944 = workspace_base + 12853248;
  tensor945 = workspace_base + 16064512;
  tensor946 = workspace_base + 16065536;
  tensor947 = workspace_base + 16066560;
  tensor948 = workspace_base + 16067584;
  tensor949 = workspace_base + 19278848;

  // Data & mask generation
  generate_data(tensor934, 3211264);
  generate_data(tensor938, 3211264);
  generate_data(tensor939, 3211264);
  generate_data(tensor955, 3211264);
  generate_data(tensor940, 2048);
  generate_data(tensor941, 2048);
  generate_data(tensor950, 1024);
  generate_data(tensor951, 1024);
  generate_data(tensor942, 1024);
  generate_data(tensor943, 1024);
  generate_data(tensor944, 3211264);
  generate_data(tensor945, 1024);
  generate_data(tensor946, 1024);
  generate_data(tensor947, 1024);
  generate_data(tensor948, 3211264);
  generate_data(tensor949, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
