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
float *tensor226, *tensor230, *tensor231, *tensor247, *tensor232, *tensor233, *tensor242, *tensor243;
float *tensor234, *tensor235, *tensor236, *tensor237, *tensor238, *tensor239, *tensor240, *tensor241;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // BatchNorm Size: 1024
  apply_grad<<<64, 0>>>(tensor232, tensor233, 1024);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("376 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 308322304);
  
  // Weight addr assignment
  tensor226 = workspace_base + 0;
  tensor230 = workspace_base + 12845056;
  tensor231 = workspace_base + 25690112;
  tensor247 = workspace_base + 38535168;
  tensor232 = workspace_base + 51380224;
  tensor233 = workspace_base + 51381248;
  tensor242 = workspace_base + 51382272;
  tensor243 = workspace_base + 51383296;
  tensor234 = workspace_base + 51384320;
  tensor235 = workspace_base + 51385344;
  tensor236 = workspace_base + 51386368;
  tensor237 = workspace_base + 64231424;
  tensor238 = workspace_base + 64232448;
  tensor239 = workspace_base + 64233472;
  tensor240 = workspace_base + 64234496;
  tensor241 = workspace_base + 77079552;

  // Data & mask generation
  generate_data(tensor226, 12845056);
  generate_data(tensor230, 12845056);
  generate_data(tensor231, 12845056);
  generate_data(tensor247, 12845056);
  generate_data(tensor232, 1024);
  generate_data(tensor233, 1024);
  generate_data(tensor242, 1024);
  generate_data(tensor243, 1024);
  generate_data(tensor234, 1024);
  generate_data(tensor235, 1024);
  generate_data(tensor236, 12845056);
  generate_data(tensor237, 1024);
  generate_data(tensor238, 1024);
  generate_data(tensor239, 1024);
  generate_data(tensor240, 12845056);
  generate_data(tensor241, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
