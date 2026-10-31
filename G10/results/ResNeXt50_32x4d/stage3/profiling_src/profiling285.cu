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
float *tensor524, *tensor528, *tensor529, *tensor545, *tensor530, *tensor531, *tensor540, *tensor541;
float *tensor532, *tensor533, *tensor534, *tensor535, *tensor536, *tensor537, *tensor538, *tensor539;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][512][14][14], V1:[64][512][14][14], V2:[512], Gamma_beta:[1024], D_v1:[64][512][14][14], D_v2:[512], D_var:[512], D_mu:[512], D_input:[64][512][14][14], D_gamma_beta:[1024]
  backwardPass_BatchNorm2d_l75<<<64, 0>>>((float (*)[512][14][14]) tensor545, (float (*)[512][14][14]) tensor534, (float (*)) tensor535, (float (*)) tensor530, (float (*)[512][14][14]) tensor538, (float (*)) tensor539, (float (*)) tensor537, (float (*)) tensor536, (float (*)[512][14][14]) tensor529, (float (*)) tensor531);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("285 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 154181632);
  
  // Weight addr assignment
  tensor524 = workspace_base + 0;
  tensor528 = workspace_base + 6422528;
  tensor529 = workspace_base + 12845056;
  tensor545 = workspace_base + 19267584;
  tensor530 = workspace_base + 25690112;
  tensor531 = workspace_base + 25691136;
  tensor540 = workspace_base + 25692160;
  tensor541 = workspace_base + 25693184;
  tensor532 = workspace_base + 25694208;
  tensor533 = workspace_base + 25695232;
  tensor534 = workspace_base + 25696256;
  tensor535 = workspace_base + 32118784;
  tensor536 = workspace_base + 32119808;
  tensor537 = workspace_base + 32120832;
  tensor538 = workspace_base + 32121856;
  tensor539 = workspace_base + 38544384;

  // Data & mask generation
  generate_data(tensor524, 6422528);
  generate_data(tensor528, 6422528);
  generate_data(tensor529, 6422528);
  generate_data(tensor545, 6422528);
  generate_data(tensor530, 1024);
  generate_data(tensor531, 1024);
  generate_data(tensor540, 1024);
  generate_data(tensor541, 1024);
  generate_data(tensor532, 1024);
  generate_data(tensor533, 1024);
  generate_data(tensor534, 6422528);
  generate_data(tensor535, 1024);
  generate_data(tensor536, 1024);
  generate_data(tensor537, 1024);
  generate_data(tensor538, 6422528);
  generate_data(tensor539, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
