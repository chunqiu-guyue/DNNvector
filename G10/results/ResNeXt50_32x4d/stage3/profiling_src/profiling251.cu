#include "../include/declaration4.cuh"
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
float *tensor636, *tensor640, *tensor641, *tensor657, *tensor642, *tensor643, *tensor652, *tensor653;
float *tensor644, *tensor645, *tensor646, *tensor647, *tensor648, *tensor649, *tensor650, *tensor651;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][512][14][14], V1:[64][512][14][14], V2:[512], Gamma_beta:[1024], D_v1:[64][512][14][14], D_v2:[512], D_var:[512], D_mu:[512], D_input:[64][512][14][14], D_gamma_beta:[1024]
  backwardPass_BatchNorm2d_l91<<<64, 0>>>((float (*)[512][14][14]) tensor657, (float (*)[512][14][14]) tensor646, (float (*)) tensor647, (float (*)) tensor642, (float (*)[512][14][14]) tensor650, (float (*)) tensor651, (float (*)) tensor649, (float (*)) tensor648, (float (*)[512][14][14]) tensor641, (float (*)) tensor643);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("251 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 154181632);
  
  // Weight addr assignment
  tensor636 = workspace_base + 0;
  tensor640 = workspace_base + 6422528;
  tensor641 = workspace_base + 12845056;
  tensor657 = workspace_base + 19267584;
  tensor642 = workspace_base + 25690112;
  tensor643 = workspace_base + 25691136;
  tensor652 = workspace_base + 25692160;
  tensor653 = workspace_base + 25693184;
  tensor644 = workspace_base + 25694208;
  tensor645 = workspace_base + 25695232;
  tensor646 = workspace_base + 25696256;
  tensor647 = workspace_base + 32118784;
  tensor648 = workspace_base + 32119808;
  tensor649 = workspace_base + 32120832;
  tensor650 = workspace_base + 32121856;
  tensor651 = workspace_base + 38544384;

  // Data & mask generation
  generate_data(tensor636, 6422528);
  generate_data(tensor640, 6422528);
  generate_data(tensor641, 6422528);
  generate_data(tensor657, 6422528);
  generate_data(tensor642, 1024);
  generate_data(tensor643, 1024);
  generate_data(tensor652, 1024);
  generate_data(tensor653, 1024);
  generate_data(tensor644, 1024);
  generate_data(tensor645, 1024);
  generate_data(tensor646, 6422528);
  generate_data(tensor647, 1024);
  generate_data(tensor648, 1024);
  generate_data(tensor649, 1024);
  generate_data(tensor650, 6422528);
  generate_data(tensor651, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
