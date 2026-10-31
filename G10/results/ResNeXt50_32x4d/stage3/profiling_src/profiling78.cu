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
float *tensor542, *tensor546, *tensor547, *tensor563, *tensor548, *tensor549, *tensor558, *tensor559;
float *tensor550, *tensor551, *tensor552, *tensor553, *tensor554, *tensor555, *tensor556, *tensor557;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][512][14][14], Gamma_beta:[1024], Running_m:[512], Running_v:[512], Mu:[512], Var:[512], V1:[64][512][14][14], V2:[512], Output:[64][512][14][14]
  forwardPass_BatchNorm2d_l77<<<64, 0>>>((float (*)[512][14][14]) tensor542, (float (*)) tensor548, (float (*)) tensor558, (float (*)) tensor559, (float (*)) tensor550, (float (*)) tensor551, (float (*)[512][14][14]) tensor552, (float (*)) tensor553, (float (*)[512][14][14]) tensor546);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("078 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 154181632);
  
  // Weight addr assignment
  tensor542 = workspace_base + 0;
  tensor546 = workspace_base + 6422528;
  tensor547 = workspace_base + 12845056;
  tensor563 = workspace_base + 19267584;
  tensor548 = workspace_base + 25690112;
  tensor549 = workspace_base + 25691136;
  tensor558 = workspace_base + 25692160;
  tensor559 = workspace_base + 25693184;
  tensor550 = workspace_base + 25694208;
  tensor551 = workspace_base + 25695232;
  tensor552 = workspace_base + 25696256;
  tensor553 = workspace_base + 32118784;
  tensor554 = workspace_base + 32119808;
  tensor555 = workspace_base + 32120832;
  tensor556 = workspace_base + 32121856;
  tensor557 = workspace_base + 38544384;

  // Data & mask generation
  generate_data(tensor542, 6422528);
  generate_data(tensor546, 6422528);
  generate_data(tensor547, 6422528);
  generate_data(tensor563, 6422528);
  generate_data(tensor548, 1024);
  generate_data(tensor549, 1024);
  generate_data(tensor558, 1024);
  generate_data(tensor559, 1024);
  generate_data(tensor550, 1024);
  generate_data(tensor551, 1024);
  generate_data(tensor552, 6422528);
  generate_data(tensor553, 1024);
  generate_data(tensor554, 1024);
  generate_data(tensor555, 1024);
  generate_data(tensor556, 6422528);
  generate_data(tensor557, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
