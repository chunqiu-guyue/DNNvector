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
float *tensor318, *tensor322, *tensor323, *tensor336, *tensor324, *tensor325, *tensor334, *tensor335;
float *tensor326, *tensor327, *tensor328, *tensor329, *tensor330, *tensor331, *tensor332, *tensor333;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][512][28][28], Gamma_beta:[1024], Running_m:[512], Running_v:[512], Mu:[512], Var:[512], V1:[64][512][28][28], V2:[512], Output:[64][512][28][28]
  forwardPass_BatchNorm2d_l45<<<64, 0>>>((float (*)[512][28][28]) tensor318, (float (*)) tensor324, (float (*)) tensor334, (float (*)) tensor335, (float (*)) tensor326, (float (*)) tensor327, (float (*)[512][28][28]) tensor328, (float (*)) tensor329, (float (*)[512][28][28]) tensor322);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("046 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 616603648);
  
  // Weight addr assignment
  tensor318 = workspace_base + 0;
  tensor322 = workspace_base + 25690112;
  tensor323 = workspace_base + 51380224;
  tensor336 = workspace_base + 77070336;
  tensor324 = workspace_base + 102760448;
  tensor325 = workspace_base + 102761472;
  tensor334 = workspace_base + 102762496;
  tensor335 = workspace_base + 102763520;
  tensor326 = workspace_base + 102764544;
  tensor327 = workspace_base + 102765568;
  tensor328 = workspace_base + 102766592;
  tensor329 = workspace_base + 128456704;
  tensor330 = workspace_base + 128457728;
  tensor331 = workspace_base + 128458752;
  tensor332 = workspace_base + 128459776;
  tensor333 = workspace_base + 154149888;

  // Data & mask generation
  generate_data(tensor318, 25690112);
  generate_data(tensor322, 25690112);
  generate_data(tensor323, 25690112);
  generate_data(tensor336, 25690112);
  generate_data(tensor324, 1024);
  generate_data(tensor325, 1024);
  generate_data(tensor334, 1024);
  generate_data(tensor335, 1024);
  generate_data(tensor326, 1024);
  generate_data(tensor327, 1024);
  generate_data(tensor328, 25690112);
  generate_data(tensor329, 1024);
  generate_data(tensor330, 1024);
  generate_data(tensor331, 1024);
  generate_data(tensor332, 25690112);
  generate_data(tensor333, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
