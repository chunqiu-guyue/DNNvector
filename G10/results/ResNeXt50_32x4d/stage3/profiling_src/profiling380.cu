#include "../include/declaration5.cuh"
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
float *tensor208, *tensor212, *tensor213, *tensor229, *tensor214, *tensor215, *tensor224, *tensor225;
float *tensor216, *tensor217, *tensor218, *tensor219, *tensor220, *tensor221, *tensor222, *tensor223;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][256][56][56], V1:[64][256][56][56], V2:[256], Gamma_beta:[512], D_v1:[64][256][56][56], D_v2:[256], D_var:[256], D_mu:[256], D_input:[64][256][56][56], D_gamma_beta:[512]
  backwardPass_BatchNorm2d_l31<<<64, 0>>>((float (*)[256][56][56]) tensor229, (float (*)[256][56][56]) tensor218, (float (*)) tensor219, (float (*)) tensor214, (float (*)[256][56][56]) tensor222, (float (*)) tensor223, (float (*)) tensor221, (float (*)) tensor220, (float (*)[256][56][56]) tensor213, (float (*)) tensor215);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("380 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 1233166336);
  
  // Weight addr assignment
  tensor208 = workspace_base + 0;
  tensor212 = workspace_base + 51380224;
  tensor213 = workspace_base + 102760448;
  tensor229 = workspace_base + 154140672;
  tensor214 = workspace_base + 205520896;
  tensor215 = workspace_base + 205521920;
  tensor224 = workspace_base + 205522944;
  tensor225 = workspace_base + 205523968;
  tensor216 = workspace_base + 205524992;
  tensor217 = workspace_base + 205526016;
  tensor218 = workspace_base + 205527040;
  tensor219 = workspace_base + 256907264;
  tensor220 = workspace_base + 256908288;
  tensor221 = workspace_base + 256909312;
  tensor222 = workspace_base + 256910336;
  tensor223 = workspace_base + 308290560;

  // Data & mask generation
  generate_data(tensor208, 51380224);
  generate_data(tensor212, 51380224);
  generate_data(tensor213, 51380224);
  generate_data(tensor229, 51380224);
  generate_data(tensor214, 1024);
  generate_data(tensor215, 1024);
  generate_data(tensor224, 1024);
  generate_data(tensor225, 1024);
  generate_data(tensor216, 1024);
  generate_data(tensor217, 1024);
  generate_data(tensor218, 51380224);
  generate_data(tensor219, 1024);
  generate_data(tensor220, 1024);
  generate_data(tensor221, 1024);
  generate_data(tensor222, 51380224);
  generate_data(tensor223, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
