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
float *tensor896, *tensor900, *tensor901, *tensor917, *tensor902, *tensor903, *tensor912, *tensor913;
float *tensor904, *tensor905, *tensor906, *tensor907, *tensor908, *tensor909, *tensor910, *tensor911;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // BatchNorm Size: 2048
  apply_grad<<<64, 0>>>(tensor902, tensor903, 2048);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("174 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 77119488);
  
  // Weight addr assignment
  tensor896 = workspace_base + 0;
  tensor900 = workspace_base + 3211264;
  tensor901 = workspace_base + 6422528;
  tensor917 = workspace_base + 9633792;
  tensor902 = workspace_base + 12845056;
  tensor903 = workspace_base + 12847104;
  tensor912 = workspace_base + 12849152;
  tensor913 = workspace_base + 12850176;
  tensor904 = workspace_base + 12851200;
  tensor905 = workspace_base + 12852224;
  tensor906 = workspace_base + 12853248;
  tensor907 = workspace_base + 16064512;
  tensor908 = workspace_base + 16065536;
  tensor909 = workspace_base + 16066560;
  tensor910 = workspace_base + 16067584;
  tensor911 = workspace_base + 19278848;

  // Data & mask generation
  generate_data(tensor896, 3211264);
  generate_data(tensor900, 3211264);
  generate_data(tensor901, 3211264);
  generate_data(tensor917, 3211264);
  generate_data(tensor902, 2048);
  generate_data(tensor903, 2048);
  generate_data(tensor912, 1024);
  generate_data(tensor913, 1024);
  generate_data(tensor904, 1024);
  generate_data(tensor905, 1024);
  generate_data(tensor906, 3211264);
  generate_data(tensor907, 1024);
  generate_data(tensor908, 1024);
  generate_data(tensor909, 1024);
  generate_data(tensor910, 3211264);
  generate_data(tensor911, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
