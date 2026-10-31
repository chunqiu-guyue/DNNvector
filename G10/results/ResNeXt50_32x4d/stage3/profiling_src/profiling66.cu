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
float *tensor450, *tensor454, *tensor455, *tensor471, *tensor456, *tensor457, *tensor466, *tensor467;
float *tensor458, *tensor459, *tensor460, *tensor461, *tensor462, *tensor463, *tensor464, *tensor465;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][512][28][28], Gamma_beta:[1024], Running_m:[512], Running_v:[512], Mu:[512], Var:[512], V1:[64][512][28][28], V2:[512], Output:[64][512][28][28]
  forwardPass_BatchNorm2d_l65<<<64, 0>>>((float (*)[512][28][28]) tensor450, (float (*)) tensor456, (float (*)) tensor466, (float (*)) tensor467, (float (*)) tensor458, (float (*)) tensor459, (float (*)[512][28][28]) tensor460, (float (*)) tensor461, (float (*)[512][28][28]) tensor454);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("066 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 616603648);
  
  // Weight addr assignment
  tensor450 = workspace_base + 0;
  tensor454 = workspace_base + 25690112;
  tensor455 = workspace_base + 51380224;
  tensor471 = workspace_base + 77070336;
  tensor456 = workspace_base + 102760448;
  tensor457 = workspace_base + 102761472;
  tensor466 = workspace_base + 102762496;
  tensor467 = workspace_base + 102763520;
  tensor458 = workspace_base + 102764544;
  tensor459 = workspace_base + 102765568;
  tensor460 = workspace_base + 102766592;
  tensor461 = workspace_base + 128456704;
  tensor462 = workspace_base + 128457728;
  tensor463 = workspace_base + 128458752;
  tensor464 = workspace_base + 128459776;
  tensor465 = workspace_base + 154149888;

  // Data & mask generation
  generate_data(tensor450, 25690112);
  generate_data(tensor454, 25690112);
  generate_data(tensor455, 25690112);
  generate_data(tensor471, 25690112);
  generate_data(tensor456, 1024);
  generate_data(tensor457, 1024);
  generate_data(tensor466, 1024);
  generate_data(tensor467, 1024);
  generate_data(tensor458, 1024);
  generate_data(tensor459, 1024);
  generate_data(tensor460, 25690112);
  generate_data(tensor461, 1024);
  generate_data(tensor462, 1024);
  generate_data(tensor463, 1024);
  generate_data(tensor464, 25690112);
  generate_data(tensor465, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
