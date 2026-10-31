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
float *tensor878, *tensor882, *tensor883, *tensor899, *tensor884, *tensor885, *tensor894, *tensor895;
float *tensor886, *tensor887, *tensor888, *tensor889, *tensor890, *tensor891, *tensor892, *tensor893;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // BatchNorm Size: 2048
  apply_grad<<<64, 0>>>(tensor884, tensor885, 2048);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("179 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 77119488);
  
  // Weight addr assignment
  tensor878 = workspace_base + 0;
  tensor882 = workspace_base + 3211264;
  tensor883 = workspace_base + 6422528;
  tensor899 = workspace_base + 9633792;
  tensor884 = workspace_base + 12845056;
  tensor885 = workspace_base + 12847104;
  tensor894 = workspace_base + 12849152;
  tensor895 = workspace_base + 12850176;
  tensor886 = workspace_base + 12851200;
  tensor887 = workspace_base + 12852224;
  tensor888 = workspace_base + 12853248;
  tensor889 = workspace_base + 16064512;
  tensor890 = workspace_base + 16065536;
  tensor891 = workspace_base + 16066560;
  tensor892 = workspace_base + 16067584;
  tensor893 = workspace_base + 19278848;

  // Data & mask generation
  generate_data(tensor878, 3211264);
  generate_data(tensor882, 3211264);
  generate_data(tensor883, 3211264);
  generate_data(tensor899, 3211264);
  generate_data(tensor884, 2048);
  generate_data(tensor885, 2048);
  generate_data(tensor894, 1024);
  generate_data(tensor895, 1024);
  generate_data(tensor886, 1024);
  generate_data(tensor887, 1024);
  generate_data(tensor888, 3211264);
  generate_data(tensor889, 1024);
  generate_data(tensor890, 1024);
  generate_data(tensor891, 1024);
  generate_data(tensor892, 3211264);
  generate_data(tensor893, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
