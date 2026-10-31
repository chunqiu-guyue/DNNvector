#include "../include/declaration3.cuh"
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
float *tensor710, *tensor714, *tensor715, *tensor731, *tensor716, *tensor717, *tensor726, *tensor727;
float *tensor718, *tensor719, *tensor720, *tensor721, *tensor722, *tensor723, *tensor724, *tensor725;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][512][14][14], V1:[64][512][14][14], V2:[512], Gamma_beta:[1024], D_v1:[64][512][14][14], D_v2:[512], D_var:[512], D_mu:[512], D_input:[64][512][14][14], D_gamma_beta:[1024]
  backwardPass_BatchNorm2d_l101<<<64, 0>>>((float (*)[512][14][14]) tensor731, (float (*)[512][14][14]) tensor720, (float (*)) tensor721, (float (*)) tensor716, (float (*)[512][14][14]) tensor724, (float (*)) tensor725, (float (*)) tensor723, (float (*)) tensor722, (float (*)[512][14][14]) tensor715, (float (*)) tensor717);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("229 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 154181632);
  
  // Weight addr assignment
  tensor710 = workspace_base + 0;
  tensor714 = workspace_base + 6422528;
  tensor715 = workspace_base + 12845056;
  tensor731 = workspace_base + 19267584;
  tensor716 = workspace_base + 25690112;
  tensor717 = workspace_base + 25691136;
  tensor726 = workspace_base + 25692160;
  tensor727 = workspace_base + 25693184;
  tensor718 = workspace_base + 25694208;
  tensor719 = workspace_base + 25695232;
  tensor720 = workspace_base + 25696256;
  tensor721 = workspace_base + 32118784;
  tensor722 = workspace_base + 32119808;
  tensor723 = workspace_base + 32120832;
  tensor724 = workspace_base + 32121856;
  tensor725 = workspace_base + 38544384;

  // Data & mask generation
  generate_data(tensor710, 6422528);
  generate_data(tensor714, 6422528);
  generate_data(tensor715, 6422528);
  generate_data(tensor731, 6422528);
  generate_data(tensor716, 1024);
  generate_data(tensor717, 1024);
  generate_data(tensor726, 1024);
  generate_data(tensor727, 1024);
  generate_data(tensor718, 1024);
  generate_data(tensor719, 1024);
  generate_data(tensor720, 6422528);
  generate_data(tensor721, 1024);
  generate_data(tensor722, 1024);
  generate_data(tensor723, 1024);
  generate_data(tensor724, 6422528);
  generate_data(tensor725, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
