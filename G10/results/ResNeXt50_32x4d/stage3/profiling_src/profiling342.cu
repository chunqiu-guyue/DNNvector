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
float *tensor338, *tensor342, *tensor343, *tensor359, *tensor344, *tensor345, *tensor354, *tensor355;
float *tensor346, *tensor347, *tensor348, *tensor349, *tensor350, *tensor351, *tensor352, *tensor353;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // BatchNorm Size: 1024
  apply_grad<<<64, 0>>>(tensor344, tensor345, 1024);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("342 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 308322304);
  
  // Weight addr assignment
  tensor338 = workspace_base + 0;
  tensor342 = workspace_base + 12845056;
  tensor343 = workspace_base + 25690112;
  tensor359 = workspace_base + 38535168;
  tensor344 = workspace_base + 51380224;
  tensor345 = workspace_base + 51381248;
  tensor354 = workspace_base + 51382272;
  tensor355 = workspace_base + 51383296;
  tensor346 = workspace_base + 51384320;
  tensor347 = workspace_base + 51385344;
  tensor348 = workspace_base + 51386368;
  tensor349 = workspace_base + 64231424;
  tensor350 = workspace_base + 64232448;
  tensor351 = workspace_base + 64233472;
  tensor352 = workspace_base + 64234496;
  tensor353 = workspace_base + 77079552;

  // Data & mask generation
  generate_data(tensor338, 12845056);
  generate_data(tensor342, 12845056);
  generate_data(tensor343, 12845056);
  generate_data(tensor359, 12845056);
  generate_data(tensor344, 1024);
  generate_data(tensor345, 1024);
  generate_data(tensor354, 1024);
  generate_data(tensor355, 1024);
  generate_data(tensor346, 1024);
  generate_data(tensor347, 1024);
  generate_data(tensor348, 12845056);
  generate_data(tensor349, 1024);
  generate_data(tensor350, 1024);
  generate_data(tensor351, 1024);
  generate_data(tensor352, 12845056);
  generate_data(tensor353, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
