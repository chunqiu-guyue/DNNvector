#include "../include/declaration2.cuh"
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
float *tensor748, *tensor752, *tensor753, *tensor769, *tensor754, *tensor755, *tensor764, *tensor765;
float *tensor756, *tensor757, *tensor758, *tensor759, *tensor760, *tensor761, *tensor762, *tensor763;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][512][14][14], Gamma_beta:[1024], Running_m:[512], Running_v:[512], Mu:[512], Var:[512], V1:[64][512][14][14], V2:[512], Output:[64][512][14][14]
  forwardPass_BatchNorm2d_l107<<<64, 0>>>((float (*)[512][14][14]) tensor748, (float (*)) tensor754, (float (*)) tensor764, (float (*)) tensor765, (float (*)) tensor756, (float (*)) tensor757, (float (*)[512][14][14]) tensor758, (float (*)) tensor759, (float (*)[512][14][14]) tensor752);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("108 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 154181632);
  
  // Weight addr assignment
  tensor748 = workspace_base + 0;
  tensor752 = workspace_base + 6422528;
  tensor753 = workspace_base + 12845056;
  tensor769 = workspace_base + 19267584;
  tensor754 = workspace_base + 25690112;
  tensor755 = workspace_base + 25691136;
  tensor764 = workspace_base + 25692160;
  tensor765 = workspace_base + 25693184;
  tensor756 = workspace_base + 25694208;
  tensor757 = workspace_base + 25695232;
  tensor758 = workspace_base + 25696256;
  tensor759 = workspace_base + 32118784;
  tensor760 = workspace_base + 32119808;
  tensor761 = workspace_base + 32120832;
  tensor762 = workspace_base + 32121856;
  tensor763 = workspace_base + 38544384;

  // Data & mask generation
  generate_data(tensor748, 6422528);
  generate_data(tensor752, 6422528);
  generate_data(tensor753, 6422528);
  generate_data(tensor769, 6422528);
  generate_data(tensor754, 1024);
  generate_data(tensor755, 1024);
  generate_data(tensor764, 1024);
  generate_data(tensor765, 1024);
  generate_data(tensor756, 1024);
  generate_data(tensor757, 1024);
  generate_data(tensor758, 6422528);
  generate_data(tensor759, 1024);
  generate_data(tensor760, 1024);
  generate_data(tensor761, 1024);
  generate_data(tensor762, 6422528);
  generate_data(tensor763, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
