#include "../include/declaration0.cuh"
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
float *tensor77, *tensor81, *tensor82, *tensor99, *tensor83, *tensor84, *tensor93, *tensor94;
float *tensor85, *tensor86, *tensor87, *tensor88, *tensor89, *tensor90, *tensor91, *tensor92;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][256][56][56], Gamma_beta:[512], Running_m:[256], Running_v:[256], Mu:[256], Var:[256], V1:[64][256][56][56], V2:[256], Output:[64][256][56][56]
  forwardPass_BatchNorm2d_l12<<<64, 0>>>((float (*)[256][56][56]) tensor77, (float (*)) tensor83, (float (*)) tensor93, (float (*)) tensor94, (float (*)) tensor85, (float (*)) tensor86, (float (*)[256][56][56]) tensor87, (float (*)) tensor88, (float (*)[256][56][56]) tensor81);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("013 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 1233166336);
  
  // Weight addr assignment
  tensor77 = workspace_base + 0;
  tensor81 = workspace_base + 51380224;
  tensor82 = workspace_base + 102760448;
  tensor99 = workspace_base + 154140672;
  tensor83 = workspace_base + 205520896;
  tensor84 = workspace_base + 205521920;
  tensor93 = workspace_base + 205522944;
  tensor94 = workspace_base + 205523968;
  tensor85 = workspace_base + 205524992;
  tensor86 = workspace_base + 205526016;
  tensor87 = workspace_base + 205527040;
  tensor88 = workspace_base + 256907264;
  tensor89 = workspace_base + 256908288;
  tensor90 = workspace_base + 256909312;
  tensor91 = workspace_base + 256910336;
  tensor92 = workspace_base + 308290560;

  // Data & mask generation
  generate_data(tensor77, 51380224);
  generate_data(tensor81, 51380224);
  generate_data(tensor82, 51380224);
  generate_data(tensor99, 51380224);
  generate_data(tensor83, 1024);
  generate_data(tensor84, 1024);
  generate_data(tensor93, 1024);
  generate_data(tensor94, 1024);
  generate_data(tensor85, 1024);
  generate_data(tensor86, 1024);
  generate_data(tensor87, 51380224);
  generate_data(tensor88, 1024);
  generate_data(tensor89, 1024);
  generate_data(tensor90, 1024);
  generate_data(tensor91, 51380224);
  generate_data(tensor92, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
