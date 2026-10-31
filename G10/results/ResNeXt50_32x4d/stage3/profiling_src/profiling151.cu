#include "../include/declaration2.cuh"
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
float *tensor970, *tensor974, *tensor975, *tensor988, *tensor976, *tensor977, *tensor986, *tensor987;
float *tensor978, *tensor979, *tensor980, *tensor981, *tensor982, *tensor983, *tensor984, *tensor985;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][2048][7][7], V1:[64][2048][7][7], V2:[2048], Gamma_beta:[4096], D_v1:[64][2048][7][7], D_v2:[2048], D_var:[2048], D_mu:[2048], D_input:[64][2048][7][7], D_gamma_beta:[4096]
  backwardPass_BatchNorm2d_l137<<<64, 0>>>((float (*)[2048][7][7]) tensor988, (float (*)[2048][7][7]) tensor980, (float (*)) tensor981, (float (*)) tensor976, (float (*)[2048][7][7]) tensor984, (float (*)) tensor985, (float (*)) tensor983, (float (*)) tensor982, (float (*)[2048][7][7]) tensor975, (float (*)) tensor977);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("151 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 154238976);
  
  // Weight addr assignment
  tensor970 = workspace_base + 0;
  tensor974 = workspace_base + 6422528;
  tensor975 = workspace_base + 12845056;
  tensor988 = workspace_base + 19267584;
  tensor976 = workspace_base + 25690112;
  tensor977 = workspace_base + 25694208;
  tensor986 = workspace_base + 25698304;
  tensor987 = workspace_base + 25700352;
  tensor978 = workspace_base + 25702400;
  tensor979 = workspace_base + 25704448;
  tensor980 = workspace_base + 25706496;
  tensor981 = workspace_base + 32129024;
  tensor982 = workspace_base + 32131072;
  tensor983 = workspace_base + 32133120;
  tensor984 = workspace_base + 32135168;
  tensor985 = workspace_base + 38557696;

  // Data & mask generation
  generate_data(tensor970, 6422528);
  generate_data(tensor974, 6422528);
  generate_data(tensor975, 6422528);
  generate_data(tensor988, 6422528);
  generate_data(tensor976, 4096);
  generate_data(tensor977, 4096);
  generate_data(tensor986, 2048);
  generate_data(tensor987, 2048);
  generate_data(tensor978, 2048);
  generate_data(tensor979, 2048);
  generate_data(tensor980, 6422528);
  generate_data(tensor981, 2048);
  generate_data(tensor982, 2048);
  generate_data(tensor983, 2048);
  generate_data(tensor984, 6422528);
  generate_data(tensor985, 2048);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
