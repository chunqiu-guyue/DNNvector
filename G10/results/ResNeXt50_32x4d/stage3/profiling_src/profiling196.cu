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
float *tensor822, *tensor826, *tensor827, *tensor843, *tensor828, *tensor829, *tensor838, *tensor839;
float *tensor830, *tensor831, *tensor832, *tensor833, *tensor834, *tensor835, *tensor836, *tensor837;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // BatchNorm Size: 2048
  apply_grad<<<64, 0>>>(tensor828, tensor829, 2048);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("196 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 77119488);
  
  // Weight addr assignment
  tensor822 = workspace_base + 0;
  tensor826 = workspace_base + 3211264;
  tensor827 = workspace_base + 6422528;
  tensor843 = workspace_base + 9633792;
  tensor828 = workspace_base + 12845056;
  tensor829 = workspace_base + 12847104;
  tensor838 = workspace_base + 12849152;
  tensor839 = workspace_base + 12850176;
  tensor830 = workspace_base + 12851200;
  tensor831 = workspace_base + 12852224;
  tensor832 = workspace_base + 12853248;
  tensor833 = workspace_base + 16064512;
  tensor834 = workspace_base + 16065536;
  tensor835 = workspace_base + 16066560;
  tensor836 = workspace_base + 16067584;
  tensor837 = workspace_base + 19278848;

  // Data & mask generation
  generate_data(tensor822, 3211264);
  generate_data(tensor826, 3211264);
  generate_data(tensor827, 3211264);
  generate_data(tensor843, 3211264);
  generate_data(tensor828, 2048);
  generate_data(tensor829, 2048);
  generate_data(tensor838, 1024);
  generate_data(tensor839, 1024);
  generate_data(tensor830, 1024);
  generate_data(tensor831, 1024);
  generate_data(tensor832, 3211264);
  generate_data(tensor833, 1024);
  generate_data(tensor834, 1024);
  generate_data(tensor835, 1024);
  generate_data(tensor836, 3211264);
  generate_data(tensor837, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
