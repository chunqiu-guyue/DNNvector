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
float *tensor804, *tensor808, *tensor809, *tensor825, *tensor810, *tensor811, *tensor820, *tensor821;
float *tensor812, *tensor813, *tensor814, *tensor815, *tensor816, *tensor817, *tensor818, *tensor819;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // BatchNorm Size: 2048
  apply_grad<<<64, 0>>>(tensor810, tensor811, 2048);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("201 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 308330496);
  
  // Weight addr assignment
  tensor804 = workspace_base + 0;
  tensor808 = workspace_base + 12845056;
  tensor809 = workspace_base + 25690112;
  tensor825 = workspace_base + 38535168;
  tensor810 = workspace_base + 51380224;
  tensor811 = workspace_base + 51382272;
  tensor820 = workspace_base + 51384320;
  tensor821 = workspace_base + 51385344;
  tensor812 = workspace_base + 51386368;
  tensor813 = workspace_base + 51387392;
  tensor814 = workspace_base + 51388416;
  tensor815 = workspace_base + 64233472;
  tensor816 = workspace_base + 64234496;
  tensor817 = workspace_base + 64235520;
  tensor818 = workspace_base + 64236544;
  tensor819 = workspace_base + 77081600;

  // Data & mask generation
  generate_data(tensor804, 12845056);
  generate_data(tensor808, 12845056);
  generate_data(tensor809, 12845056);
  generate_data(tensor825, 12845056);
  generate_data(tensor810, 2048);
  generate_data(tensor811, 2048);
  generate_data(tensor820, 1024);
  generate_data(tensor821, 1024);
  generate_data(tensor812, 1024);
  generate_data(tensor813, 1024);
  generate_data(tensor814, 12845056);
  generate_data(tensor815, 1024);
  generate_data(tensor816, 1024);
  generate_data(tensor817, 1024);
  generate_data(tensor818, 12845056);
  generate_data(tensor819, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
