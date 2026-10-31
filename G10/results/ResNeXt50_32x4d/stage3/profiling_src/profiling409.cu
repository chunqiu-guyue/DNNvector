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
float *tensor114, *tensor118, *tensor119, *tensor135, *tensor120, *tensor121, *tensor130, *tensor131;
float *tensor122, *tensor123, *tensor124, *tensor125, *tensor126, *tensor127, *tensor128, *tensor129;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][128][56][56], V1:[64][128][56][56], V2:[128], Gamma_beta:[256], D_v1:[64][128][56][56], D_v2:[128], D_var:[128], D_mu:[128], D_input:[64][128][56][56], D_gamma_beta:[256]
  backwardPass_BatchNorm2d_l17<<<64, 0>>>((float (*)[128][56][56]) tensor135, (float (*)[128][56][56]) tensor124, (float (*)) tensor125, (float (*)) tensor120, (float (*)[128][56][56]) tensor128, (float (*)) tensor129, (float (*)) tensor127, (float (*)) tensor126, (float (*)[128][56][56]) tensor119, (float (*)) tensor121);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("409 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 616603648);
  
  // Weight addr assignment
  tensor114 = workspace_base + 0;
  tensor118 = workspace_base + 25690112;
  tensor119 = workspace_base + 51380224;
  tensor135 = workspace_base + 77070336;
  tensor120 = workspace_base + 102760448;
  tensor121 = workspace_base + 102761472;
  tensor130 = workspace_base + 102762496;
  tensor131 = workspace_base + 102763520;
  tensor122 = workspace_base + 102764544;
  tensor123 = workspace_base + 102765568;
  tensor124 = workspace_base + 102766592;
  tensor125 = workspace_base + 128456704;
  tensor126 = workspace_base + 128457728;
  tensor127 = workspace_base + 128458752;
  tensor128 = workspace_base + 128459776;
  tensor129 = workspace_base + 154149888;

  // Data & mask generation
  generate_data(tensor114, 25690112);
  generate_data(tensor118, 25690112);
  generate_data(tensor119, 25690112);
  generate_data(tensor135, 25690112);
  generate_data(tensor120, 1024);
  generate_data(tensor121, 1024);
  generate_data(tensor130, 1024);
  generate_data(tensor131, 1024);
  generate_data(tensor122, 1024);
  generate_data(tensor123, 1024);
  generate_data(tensor124, 25690112);
  generate_data(tensor125, 1024);
  generate_data(tensor126, 1024);
  generate_data(tensor127, 1024);
  generate_data(tensor128, 25690112);
  generate_data(tensor129, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
