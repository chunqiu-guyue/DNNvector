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
float *tensor170, *tensor174, *tensor175, *tensor191, *tensor176, *tensor177, *tensor186, *tensor187;
float *tensor178, *tensor179, *tensor180, *tensor181, *tensor182, *tensor183, *tensor184, *tensor185;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // BatchNorm Size: 1024
  apply_grad<<<64, 0>>>(tensor176, tensor177, 1024);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("393 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 616603648);
  
  // Weight addr assignment
  tensor170 = workspace_base + 0;
  tensor174 = workspace_base + 25690112;
  tensor175 = workspace_base + 51380224;
  tensor191 = workspace_base + 77070336;
  tensor176 = workspace_base + 102760448;
  tensor177 = workspace_base + 102761472;
  tensor186 = workspace_base + 102762496;
  tensor187 = workspace_base + 102763520;
  tensor178 = workspace_base + 102764544;
  tensor179 = workspace_base + 102765568;
  tensor180 = workspace_base + 102766592;
  tensor181 = workspace_base + 128456704;
  tensor182 = workspace_base + 128457728;
  tensor183 = workspace_base + 128458752;
  tensor184 = workspace_base + 128459776;
  tensor185 = workspace_base + 154149888;

  // Data & mask generation
  generate_data(tensor170, 25690112);
  generate_data(tensor174, 25690112);
  generate_data(tensor175, 25690112);
  generate_data(tensor191, 25690112);
  generate_data(tensor176, 1024);
  generate_data(tensor177, 1024);
  generate_data(tensor186, 1024);
  generate_data(tensor187, 1024);
  generate_data(tensor178, 1024);
  generate_data(tensor179, 1024);
  generate_data(tensor180, 25690112);
  generate_data(tensor181, 1024);
  generate_data(tensor182, 1024);
  generate_data(tensor183, 1024);
  generate_data(tensor184, 25690112);
  generate_data(tensor185, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
