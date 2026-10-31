#include "../include/declaration6.cuh"
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
float *tensor1, *tensor5, *tensor6, *tensor19, *tensor7, *tensor8, *tensor17, *tensor18;
float *tensor9, *tensor10, *tensor11, *tensor12, *tensor13, *tensor14, *tensor15, *tensor16;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][64][112][112], V1:[64][64][112][112], V2:[64], Gamma_beta:[128], D_v1:[64][64][112][112], D_v2:[64], D_var:[64], D_mu:[64], D_input:[64][64][112][112], D_gamma_beta:[128]
  backwardPass_BatchNorm2d_l1<<<64, 0>>>((float (*)[64][112][112]) tensor19, (float (*)[64][112][112]) tensor11, (float (*)) tensor12, (float (*)) tensor7, (float (*)[64][112][112]) tensor15, (float (*)) tensor16, (float (*)) tensor14, (float (*)) tensor13, (float (*)[64][112][112]) tensor6, (float (*)) tensor8);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("444 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 1233166336);
  
  // Weight addr assignment
  tensor1 = workspace_base + 0;
  tensor5 = workspace_base + 51380224;
  tensor6 = workspace_base + 102760448;
  tensor19 = workspace_base + 154140672;
  tensor7 = workspace_base + 205520896;
  tensor8 = workspace_base + 205521920;
  tensor17 = workspace_base + 205522944;
  tensor18 = workspace_base + 205523968;
  tensor9 = workspace_base + 205524992;
  tensor10 = workspace_base + 205526016;
  tensor11 = workspace_base + 205527040;
  tensor12 = workspace_base + 256907264;
  tensor13 = workspace_base + 256908288;
  tensor14 = workspace_base + 256909312;
  tensor15 = workspace_base + 256910336;
  tensor16 = workspace_base + 308290560;

  // Data & mask generation
  generate_data(tensor1, 51380224);
  generate_data(tensor5, 51380224);
  generate_data(tensor6, 51380224);
  generate_data(tensor19, 51380224);
  generate_data(tensor7, 1024);
  generate_data(tensor8, 1024);
  generate_data(tensor17, 1024);
  generate_data(tensor18, 1024);
  generate_data(tensor9, 1024);
  generate_data(tensor10, 1024);
  generate_data(tensor11, 51380224);
  generate_data(tensor12, 1024);
  generate_data(tensor13, 1024);
  generate_data(tensor14, 1024);
  generate_data(tensor15, 51380224);
  generate_data(tensor16, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
