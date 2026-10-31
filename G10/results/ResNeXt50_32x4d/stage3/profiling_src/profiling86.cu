#include "../include/declaration1.cuh"
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
float *tensor598, *tensor602, *tensor603, *tensor619, *tensor604, *tensor605, *tensor614, *tensor615;
float *tensor606, *tensor607, *tensor608, *tensor609, *tensor610, *tensor611, *tensor612, *tensor613;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][512][14][14], Gamma_beta:[1024], Running_m:[512], Running_v:[512], Mu:[512], Var:[512], V1:[64][512][14][14], V2:[512], Output:[64][512][14][14]
  forwardPass_BatchNorm2d_l85<<<64, 0>>>((float (*)[512][14][14]) tensor598, (float (*)) tensor604, (float (*)) tensor614, (float (*)) tensor615, (float (*)) tensor606, (float (*)) tensor607, (float (*)[512][14][14]) tensor608, (float (*)) tensor609, (float (*)[512][14][14]) tensor602);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("086 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 154181632);
  
  // Weight addr assignment
  tensor598 = workspace_base + 0;
  tensor602 = workspace_base + 6422528;
  tensor603 = workspace_base + 12845056;
  tensor619 = workspace_base + 19267584;
  tensor604 = workspace_base + 25690112;
  tensor605 = workspace_base + 25691136;
  tensor614 = workspace_base + 25692160;
  tensor615 = workspace_base + 25693184;
  tensor606 = workspace_base + 25694208;
  tensor607 = workspace_base + 25695232;
  tensor608 = workspace_base + 25696256;
  tensor609 = workspace_base + 32118784;
  tensor610 = workspace_base + 32119808;
  tensor611 = workspace_base + 32120832;
  tensor612 = workspace_base + 32121856;
  tensor613 = workspace_base + 38544384;

  // Data & mask generation
  generate_data(tensor598, 6422528);
  generate_data(tensor602, 6422528);
  generate_data(tensor603, 6422528);
  generate_data(tensor619, 6422528);
  generate_data(tensor604, 1024);
  generate_data(tensor605, 1024);
  generate_data(tensor614, 1024);
  generate_data(tensor615, 1024);
  generate_data(tensor606, 1024);
  generate_data(tensor607, 1024);
  generate_data(tensor608, 6422528);
  generate_data(tensor609, 1024);
  generate_data(tensor610, 1024);
  generate_data(tensor611, 1024);
  generate_data(tensor612, 6422528);
  generate_data(tensor613, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
