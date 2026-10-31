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
float *tensor282, *tensor286, *tensor287, *tensor303, *tensor288, *tensor289, *tensor298, *tensor299;
float *tensor290, *tensor291, *tensor292, *tensor293, *tensor294, *tensor295, *tensor296, *tensor297;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // BatchNorm Size: 1024
  apply_grad<<<64, 0>>>(tensor288, tensor289, 1024);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("359 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 308322304);
  
  // Weight addr assignment
  tensor282 = workspace_base + 0;
  tensor286 = workspace_base + 12845056;
  tensor287 = workspace_base + 25690112;
  tensor303 = workspace_base + 38535168;
  tensor288 = workspace_base + 51380224;
  tensor289 = workspace_base + 51381248;
  tensor298 = workspace_base + 51382272;
  tensor299 = workspace_base + 51383296;
  tensor290 = workspace_base + 51384320;
  tensor291 = workspace_base + 51385344;
  tensor292 = workspace_base + 51386368;
  tensor293 = workspace_base + 64231424;
  tensor294 = workspace_base + 64232448;
  tensor295 = workspace_base + 64233472;
  tensor296 = workspace_base + 64234496;
  tensor297 = workspace_base + 77079552;

  // Data & mask generation
  generate_data(tensor282, 12845056);
  generate_data(tensor286, 12845056);
  generate_data(tensor287, 12845056);
  generate_data(tensor303, 12845056);
  generate_data(tensor288, 1024);
  generate_data(tensor289, 1024);
  generate_data(tensor298, 1024);
  generate_data(tensor299, 1024);
  generate_data(tensor290, 1024);
  generate_data(tensor291, 1024);
  generate_data(tensor292, 12845056);
  generate_data(tensor293, 1024);
  generate_data(tensor294, 1024);
  generate_data(tensor295, 1024);
  generate_data(tensor296, 12845056);
  generate_data(tensor297, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
