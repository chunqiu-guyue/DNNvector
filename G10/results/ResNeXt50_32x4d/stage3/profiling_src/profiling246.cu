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
float *tensor654, *tensor658, *tensor659, *tensor675, *tensor660, *tensor661, *tensor670, *tensor671;
float *tensor662, *tensor663, *tensor664, *tensor665, *tensor666, *tensor667, *tensor668, *tensor669;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][512][14][14], V1:[64][512][14][14], V2:[512], Gamma_beta:[1024], D_v1:[64][512][14][14], D_v2:[512], D_var:[512], D_mu:[512], D_input:[64][512][14][14], D_gamma_beta:[1024]
  backwardPass_BatchNorm2d_l93<<<64, 0>>>((float (*)[512][14][14]) tensor675, (float (*)[512][14][14]) tensor664, (float (*)) tensor665, (float (*)) tensor660, (float (*)[512][14][14]) tensor668, (float (*)) tensor669, (float (*)) tensor667, (float (*)) tensor666, (float (*)[512][14][14]) tensor659, (float (*)) tensor661);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("246 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 154181632);
  
  // Weight addr assignment
  tensor654 = workspace_base + 0;
  tensor658 = workspace_base + 6422528;
  tensor659 = workspace_base + 12845056;
  tensor675 = workspace_base + 19267584;
  tensor660 = workspace_base + 25690112;
  tensor661 = workspace_base + 25691136;
  tensor670 = workspace_base + 25692160;
  tensor671 = workspace_base + 25693184;
  tensor662 = workspace_base + 25694208;
  tensor663 = workspace_base + 25695232;
  tensor664 = workspace_base + 25696256;
  tensor665 = workspace_base + 32118784;
  tensor666 = workspace_base + 32119808;
  tensor667 = workspace_base + 32120832;
  tensor668 = workspace_base + 32121856;
  tensor669 = workspace_base + 38544384;

  // Data & mask generation
  generate_data(tensor654, 6422528);
  generate_data(tensor658, 6422528);
  generate_data(tensor659, 6422528);
  generate_data(tensor675, 6422528);
  generate_data(tensor660, 1024);
  generate_data(tensor661, 1024);
  generate_data(tensor670, 1024);
  generate_data(tensor671, 1024);
  generate_data(tensor662, 1024);
  generate_data(tensor663, 1024);
  generate_data(tensor664, 6422528);
  generate_data(tensor665, 1024);
  generate_data(tensor666, 1024);
  generate_data(tensor667, 1024);
  generate_data(tensor668, 6422528);
  generate_data(tensor669, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
