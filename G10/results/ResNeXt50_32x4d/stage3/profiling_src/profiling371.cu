#include "../include/cudadnnUtil.cuh"
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
float *tensor244, *tensor248, *tensor249, *tensor262, *tensor250, *tensor251, *tensor260, *tensor261;
float *tensor252, *tensor253, *tensor254, *tensor255, *tensor256, *tensor257, *tensor258, *tensor259;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // BatchNorm Size: 1024
  apply_grad<<<64, 0>>>(tensor250, tensor251, 1024);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("371 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 616603648);
  
  // Weight addr assignment
  tensor244 = workspace_base + 0;
  tensor248 = workspace_base + 25690112;
  tensor249 = workspace_base + 51380224;
  tensor262 = workspace_base + 77070336;
  tensor250 = workspace_base + 102760448;
  tensor251 = workspace_base + 102761472;
  tensor260 = workspace_base + 102762496;
  tensor261 = workspace_base + 102763520;
  tensor252 = workspace_base + 102764544;
  tensor253 = workspace_base + 102765568;
  tensor254 = workspace_base + 102766592;
  tensor255 = workspace_base + 128456704;
  tensor256 = workspace_base + 128457728;
  tensor257 = workspace_base + 128458752;
  tensor258 = workspace_base + 128459776;
  tensor259 = workspace_base + 154149888;

  // Data & mask generation
  generate_data(tensor244, 25690112);
  generate_data(tensor248, 25690112);
  generate_data(tensor249, 25690112);
  generate_data(tensor262, 25690112);
  generate_data(tensor250, 1024);
  generate_data(tensor251, 1024);
  generate_data(tensor260, 1024);
  generate_data(tensor261, 1024);
  generate_data(tensor252, 1024);
  generate_data(tensor253, 1024);
  generate_data(tensor254, 25690112);
  generate_data(tensor255, 1024);
  generate_data(tensor256, 1024);
  generate_data(tensor257, 1024);
  generate_data(tensor258, 25690112);
  generate_data(tensor259, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
