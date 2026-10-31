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
float *tensor96, *tensor100, *tensor101, *tensor117, *tensor102, *tensor103, *tensor112, *tensor113;
float *tensor104, *tensor105, *tensor106, *tensor107, *tensor108, *tensor109, *tensor110, *tensor111;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][128][56][56], V1:[64][128][56][56], V2:[128], Gamma_beta:[256], D_v1:[64][128][56][56], D_v2:[128], D_var:[128], D_mu:[128], D_input:[64][128][56][56], D_gamma_beta:[256]
  backwardPass_BatchNorm2d_l15<<<64, 0>>>((float (*)[128][56][56]) tensor117, (float (*)[128][56][56]) tensor106, (float (*)) tensor107, (float (*)) tensor102, (float (*)[128][56][56]) tensor110, (float (*)) tensor111, (float (*)) tensor109, (float (*)) tensor108, (float (*)[128][56][56]) tensor101, (float (*)) tensor103);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("414 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 616603648);
  
  // Weight addr assignment
  tensor96 = workspace_base + 0;
  tensor100 = workspace_base + 25690112;
  tensor101 = workspace_base + 51380224;
  tensor117 = workspace_base + 77070336;
  tensor102 = workspace_base + 102760448;
  tensor103 = workspace_base + 102761472;
  tensor112 = workspace_base + 102762496;
  tensor113 = workspace_base + 102763520;
  tensor104 = workspace_base + 102764544;
  tensor105 = workspace_base + 102765568;
  tensor106 = workspace_base + 102766592;
  tensor107 = workspace_base + 128456704;
  tensor108 = workspace_base + 128457728;
  tensor109 = workspace_base + 128458752;
  tensor110 = workspace_base + 128459776;
  tensor111 = workspace_base + 154149888;

  // Data & mask generation
  generate_data(tensor96, 25690112);
  generate_data(tensor100, 25690112);
  generate_data(tensor101, 25690112);
  generate_data(tensor117, 25690112);
  generate_data(tensor102, 1024);
  generate_data(tensor103, 1024);
  generate_data(tensor112, 1024);
  generate_data(tensor113, 1024);
  generate_data(tensor104, 1024);
  generate_data(tensor105, 1024);
  generate_data(tensor106, 25690112);
  generate_data(tensor107, 1024);
  generate_data(tensor108, 1024);
  generate_data(tensor109, 1024);
  generate_data(tensor110, 25690112);
  generate_data(tensor111, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
