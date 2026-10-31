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
float *tensor692, *tensor696, *tensor697, *tensor713, *tensor698, *tensor699, *tensor708, *tensor709;
float *tensor700, *tensor701, *tensor702, *tensor703, *tensor704, *tensor705, *tensor706, *tensor707;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][512][14][14], V1:[64][512][14][14], V2:[512], Gamma_beta:[1024], D_v1:[64][512][14][14], D_v2:[512], D_var:[512], D_mu:[512], D_input:[64][512][14][14], D_gamma_beta:[1024]
  backwardPass_BatchNorm2d_l99<<<64, 0>>>((float (*)[512][14][14]) tensor713, (float (*)[512][14][14]) tensor702, (float (*)) tensor703, (float (*)) tensor698, (float (*)[512][14][14]) tensor706, (float (*)) tensor707, (float (*)) tensor705, (float (*)) tensor704, (float (*)[512][14][14]) tensor697, (float (*)) tensor699);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("234 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 154181632);
  
  // Weight addr assignment
  tensor692 = workspace_base + 0;
  tensor696 = workspace_base + 6422528;
  tensor697 = workspace_base + 12845056;
  tensor713 = workspace_base + 19267584;
  tensor698 = workspace_base + 25690112;
  tensor699 = workspace_base + 25691136;
  tensor708 = workspace_base + 25692160;
  tensor709 = workspace_base + 25693184;
  tensor700 = workspace_base + 25694208;
  tensor701 = workspace_base + 25695232;
  tensor702 = workspace_base + 25696256;
  tensor703 = workspace_base + 32118784;
  tensor704 = workspace_base + 32119808;
  tensor705 = workspace_base + 32120832;
  tensor706 = workspace_base + 32121856;
  tensor707 = workspace_base + 38544384;

  // Data & mask generation
  generate_data(tensor692, 6422528);
  generate_data(tensor696, 6422528);
  generate_data(tensor697, 6422528);
  generate_data(tensor713, 6422528);
  generate_data(tensor698, 1024);
  generate_data(tensor699, 1024);
  generate_data(tensor708, 1024);
  generate_data(tensor709, 1024);
  generate_data(tensor700, 1024);
  generate_data(tensor701, 1024);
  generate_data(tensor702, 6422528);
  generate_data(tensor703, 1024);
  generate_data(tensor704, 1024);
  generate_data(tensor705, 1024);
  generate_data(tensor706, 6422528);
  generate_data(tensor707, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
