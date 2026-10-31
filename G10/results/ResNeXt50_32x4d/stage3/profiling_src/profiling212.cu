#include "../include/declaration3.cuh"
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
float *tensor766, *tensor770, *tensor771, *tensor787, *tensor772, *tensor773, *tensor782, *tensor783;
float *tensor774, *tensor775, *tensor776, *tensor777, *tensor778, *tensor779, *tensor780, *tensor781;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][512][14][14], V1:[64][512][14][14], V2:[512], Gamma_beta:[1024], D_v1:[64][512][14][14], D_v2:[512], D_var:[512], D_mu:[512], D_input:[64][512][14][14], D_gamma_beta:[1024]
  backwardPass_BatchNorm2d_l109<<<64, 0>>>((float (*)[512][14][14]) tensor787, (float (*)[512][14][14]) tensor776, (float (*)) tensor777, (float (*)) tensor772, (float (*)[512][14][14]) tensor780, (float (*)) tensor781, (float (*)) tensor779, (float (*)) tensor778, (float (*)[512][14][14]) tensor771, (float (*)) tensor773);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("212 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 154181632);
  
  // Weight addr assignment
  tensor766 = workspace_base + 0;
  tensor770 = workspace_base + 6422528;
  tensor771 = workspace_base + 12845056;
  tensor787 = workspace_base + 19267584;
  tensor772 = workspace_base + 25690112;
  tensor773 = workspace_base + 25691136;
  tensor782 = workspace_base + 25692160;
  tensor783 = workspace_base + 25693184;
  tensor774 = workspace_base + 25694208;
  tensor775 = workspace_base + 25695232;
  tensor776 = workspace_base + 25696256;
  tensor777 = workspace_base + 32118784;
  tensor778 = workspace_base + 32119808;
  tensor779 = workspace_base + 32120832;
  tensor780 = workspace_base + 32121856;
  tensor781 = workspace_base + 38544384;

  // Data & mask generation
  generate_data(tensor766, 6422528);
  generate_data(tensor770, 6422528);
  generate_data(tensor771, 6422528);
  generate_data(tensor787, 6422528);
  generate_data(tensor772, 1024);
  generate_data(tensor773, 1024);
  generate_data(tensor782, 1024);
  generate_data(tensor783, 1024);
  generate_data(tensor774, 1024);
  generate_data(tensor775, 1024);
  generate_data(tensor776, 6422528);
  generate_data(tensor777, 1024);
  generate_data(tensor778, 1024);
  generate_data(tensor779, 1024);
  generate_data(tensor780, 6422528);
  generate_data(tensor781, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
