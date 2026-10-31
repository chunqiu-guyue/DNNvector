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
float *tensor152, *tensor156, *tensor157, *tensor173, *tensor158, *tensor159, *tensor168, *tensor169;
float *tensor160, *tensor161, *tensor162, *tensor163, *tensor164, *tensor165, *tensor166, *tensor167;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][128][56][56], Gamma_beta:[256], Running_m:[128], Running_v:[128], Mu:[128], Var:[128], V1:[64][128][56][56], V2:[128], Output:[64][128][56][56]
  forwardPass_BatchNorm2d_l23<<<64, 0>>>((float (*)[128][56][56]) tensor152, (float (*)) tensor158, (float (*)) tensor168, (float (*)) tensor169, (float (*)) tensor160, (float (*)) tensor161, (float (*)[128][56][56]) tensor162, (float (*)) tensor163, (float (*)[128][56][56]) tensor156);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("024 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 616603648);
  
  // Weight addr assignment
  tensor152 = workspace_base + 0;
  tensor156 = workspace_base + 25690112;
  tensor157 = workspace_base + 51380224;
  tensor173 = workspace_base + 77070336;
  tensor158 = workspace_base + 102760448;
  tensor159 = workspace_base + 102761472;
  tensor168 = workspace_base + 102762496;
  tensor169 = workspace_base + 102763520;
  tensor160 = workspace_base + 102764544;
  tensor161 = workspace_base + 102765568;
  tensor162 = workspace_base + 102766592;
  tensor163 = workspace_base + 128456704;
  tensor164 = workspace_base + 128457728;
  tensor165 = workspace_base + 128458752;
  tensor166 = workspace_base + 128459776;
  tensor167 = workspace_base + 154149888;

  // Data & mask generation
  generate_data(tensor152, 25690112);
  generate_data(tensor156, 25690112);
  generate_data(tensor157, 25690112);
  generate_data(tensor173, 25690112);
  generate_data(tensor158, 1024);
  generate_data(tensor159, 1024);
  generate_data(tensor168, 1024);
  generate_data(tensor169, 1024);
  generate_data(tensor160, 1024);
  generate_data(tensor161, 1024);
  generate_data(tensor162, 25690112);
  generate_data(tensor163, 1024);
  generate_data(tensor164, 1024);
  generate_data(tensor165, 1024);
  generate_data(tensor166, 25690112);
  generate_data(tensor167, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
