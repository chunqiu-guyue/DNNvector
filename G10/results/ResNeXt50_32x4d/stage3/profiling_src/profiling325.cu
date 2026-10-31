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
float *tensor394, *tensor398, *tensor399, *tensor415, *tensor400, *tensor401, *tensor410, *tensor411;
float *tensor402, *tensor403, *tensor404, *tensor405, *tensor406, *tensor407, *tensor408, *tensor409;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // BatchNorm Size: 1024
  apply_grad<<<64, 0>>>(tensor400, tensor401, 1024);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("325 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 308322304);
  
  // Weight addr assignment
  tensor394 = workspace_base + 0;
  tensor398 = workspace_base + 12845056;
  tensor399 = workspace_base + 25690112;
  tensor415 = workspace_base + 38535168;
  tensor400 = workspace_base + 51380224;
  tensor401 = workspace_base + 51381248;
  tensor410 = workspace_base + 51382272;
  tensor411 = workspace_base + 51383296;
  tensor402 = workspace_base + 51384320;
  tensor403 = workspace_base + 51385344;
  tensor404 = workspace_base + 51386368;
  tensor405 = workspace_base + 64231424;
  tensor406 = workspace_base + 64232448;
  tensor407 = workspace_base + 64233472;
  tensor408 = workspace_base + 64234496;
  tensor409 = workspace_base + 77079552;

  // Data & mask generation
  generate_data(tensor394, 12845056);
  generate_data(tensor398, 12845056);
  generate_data(tensor399, 12845056);
  generate_data(tensor415, 12845056);
  generate_data(tensor400, 1024);
  generate_data(tensor401, 1024);
  generate_data(tensor410, 1024);
  generate_data(tensor411, 1024);
  generate_data(tensor402, 1024);
  generate_data(tensor403, 1024);
  generate_data(tensor404, 12845056);
  generate_data(tensor405, 1024);
  generate_data(tensor406, 1024);
  generate_data(tensor407, 1024);
  generate_data(tensor408, 12845056);
  generate_data(tensor409, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
