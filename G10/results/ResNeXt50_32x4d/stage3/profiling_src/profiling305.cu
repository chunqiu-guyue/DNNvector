#include "../include/declaration4.cuh"
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
float *tensor454, *tensor468, *tensor469, *tensor471, *tensor473, *tensor470;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_in:[64][512][28][28], D_out:[64][512][14][14], Weight:[512][512][3][3]
  backwardPass_conv2d_Input_l66<<<64, 0>>>((float (*)[512][28][28]) tensor471, (float (*)[512][14][14]) tensor473, (float (*)[512][3][3]) tensor469);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("305 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 257490944);
  
  // Weight addr assignment
  tensor454 = workspace_base + 0;
  tensor468 = workspace_base + 25690112;
  tensor469 = workspace_base + 32112640;
  tensor471 = workspace_base + 32186368;
  tensor473 = workspace_base + 57876480;
  tensor470 = workspace_base + 64299008;

  // Data & mask generation
  generate_data(tensor454, 25690112);
  generate_data(tensor468, 6422528);
  generate_data(tensor469, 73728);
  generate_data(tensor471, 25690112);
  generate_data(tensor473, 6422528);
  generate_data(tensor470, 73728);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
