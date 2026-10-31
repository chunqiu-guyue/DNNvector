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
float *tensor580, *tensor584, *tensor585, *tensor601, *tensor586, *tensor587, *tensor596, *tensor597;
float *tensor588, *tensor589, *tensor590, *tensor591, *tensor592, *tensor593, *tensor594, *tensor595;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][512][14][14], V1:[64][512][14][14], V2:[512], Gamma_beta:[1024], D_v1:[64][512][14][14], D_v2:[512], D_var:[512], D_mu:[512], D_input:[64][512][14][14], D_gamma_beta:[1024]
  backwardPass_BatchNorm2d_l83<<<64, 0>>>((float (*)[512][14][14]) tensor601, (float (*)[512][14][14]) tensor590, (float (*)) tensor591, (float (*)) tensor586, (float (*)[512][14][14]) tensor594, (float (*)) tensor595, (float (*)) tensor593, (float (*)) tensor592, (float (*)[512][14][14]) tensor585, (float (*)) tensor587);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("268 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 154181632);
  
  // Weight addr assignment
  tensor580 = workspace_base + 0;
  tensor584 = workspace_base + 6422528;
  tensor585 = workspace_base + 12845056;
  tensor601 = workspace_base + 19267584;
  tensor586 = workspace_base + 25690112;
  tensor587 = workspace_base + 25691136;
  tensor596 = workspace_base + 25692160;
  tensor597 = workspace_base + 25693184;
  tensor588 = workspace_base + 25694208;
  tensor589 = workspace_base + 25695232;
  tensor590 = workspace_base + 25696256;
  tensor591 = workspace_base + 32118784;
  tensor592 = workspace_base + 32119808;
  tensor593 = workspace_base + 32120832;
  tensor594 = workspace_base + 32121856;
  tensor595 = workspace_base + 38544384;

  // Data & mask generation
  generate_data(tensor580, 6422528);
  generate_data(tensor584, 6422528);
  generate_data(tensor585, 6422528);
  generate_data(tensor601, 6422528);
  generate_data(tensor586, 1024);
  generate_data(tensor587, 1024);
  generate_data(tensor596, 1024);
  generate_data(tensor597, 1024);
  generate_data(tensor588, 1024);
  generate_data(tensor589, 1024);
  generate_data(tensor590, 6422528);
  generate_data(tensor591, 1024);
  generate_data(tensor592, 1024);
  generate_data(tensor593, 1024);
  generate_data(tensor594, 6422528);
  generate_data(tensor595, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
