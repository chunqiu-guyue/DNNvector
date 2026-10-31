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
float *tensor58, *tensor62, *tensor63, *tensor76, *tensor64, *tensor65, *tensor74, *tensor75;
float *tensor66, *tensor67, *tensor68, *tensor69, *tensor70, *tensor71, *tensor72, *tensor73;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][256][56][56], Gamma_beta:[512], Running_m:[256], Running_v:[256], Mu:[256], Var:[256], V1:[64][256][56][56], V2:[256], Output:[64][256][56][56]
  forwardPass_BatchNorm2d_l9<<<64, 0>>>((float (*)[256][56][56]) tensor58, (float (*)) tensor64, (float (*)) tensor74, (float (*)) tensor75, (float (*)) tensor66, (float (*)) tensor67, (float (*)[256][56][56]) tensor68, (float (*)) tensor69, (float (*)[256][56][56]) tensor62);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("010 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 1233166336);
  
  // Weight addr assignment
  tensor58 = workspace_base + 0;
  tensor62 = workspace_base + 51380224;
  tensor63 = workspace_base + 102760448;
  tensor76 = workspace_base + 154140672;
  tensor64 = workspace_base + 205520896;
  tensor65 = workspace_base + 205521920;
  tensor74 = workspace_base + 205522944;
  tensor75 = workspace_base + 205523968;
  tensor66 = workspace_base + 205524992;
  tensor67 = workspace_base + 205526016;
  tensor68 = workspace_base + 205527040;
  tensor69 = workspace_base + 256907264;
  tensor70 = workspace_base + 256908288;
  tensor71 = workspace_base + 256909312;
  tensor72 = workspace_base + 256910336;
  tensor73 = workspace_base + 308290560;

  // Data & mask generation
  generate_data(tensor58, 51380224);
  generate_data(tensor62, 51380224);
  generate_data(tensor63, 51380224);
  generate_data(tensor76, 51380224);
  generate_data(tensor64, 1024);
  generate_data(tensor65, 1024);
  generate_data(tensor74, 1024);
  generate_data(tensor75, 1024);
  generate_data(tensor66, 1024);
  generate_data(tensor67, 1024);
  generate_data(tensor68, 51380224);
  generate_data(tensor69, 1024);
  generate_data(tensor70, 1024);
  generate_data(tensor71, 1024);
  generate_data(tensor72, 51380224);
  generate_data(tensor73, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
