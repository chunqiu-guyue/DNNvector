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
float *tensor560, *tensor564, *tensor565, *tensor578, *tensor566, *tensor567, *tensor576, *tensor577;
float *tensor568, *tensor569, *tensor570, *tensor571, *tensor572, *tensor573, *tensor574, *tensor575;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // BatchNorm Size: 2048
  apply_grad<<<64, 0>>>(tensor566, tensor567, 2048);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("276 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 308330496);
  
  // Weight addr assignment
  tensor560 = workspace_base + 0;
  tensor564 = workspace_base + 12845056;
  tensor565 = workspace_base + 25690112;
  tensor578 = workspace_base + 38535168;
  tensor566 = workspace_base + 51380224;
  tensor567 = workspace_base + 51382272;
  tensor576 = workspace_base + 51384320;
  tensor577 = workspace_base + 51385344;
  tensor568 = workspace_base + 51386368;
  tensor569 = workspace_base + 51387392;
  tensor570 = workspace_base + 51388416;
  tensor571 = workspace_base + 64233472;
  tensor572 = workspace_base + 64234496;
  tensor573 = workspace_base + 64235520;
  tensor574 = workspace_base + 64236544;
  tensor575 = workspace_base + 77081600;

  // Data & mask generation
  generate_data(tensor560, 12845056);
  generate_data(tensor564, 12845056);
  generate_data(tensor565, 12845056);
  generate_data(tensor578, 12845056);
  generate_data(tensor566, 2048);
  generate_data(tensor567, 2048);
  generate_data(tensor576, 1024);
  generate_data(tensor577, 1024);
  generate_data(tensor568, 1024);
  generate_data(tensor569, 1024);
  generate_data(tensor570, 12845056);
  generate_data(tensor571, 1024);
  generate_data(tensor572, 1024);
  generate_data(tensor573, 1024);
  generate_data(tensor574, 12845056);
  generate_data(tensor575, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
