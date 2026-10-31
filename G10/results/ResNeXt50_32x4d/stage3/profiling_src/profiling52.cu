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
float *tensor356, *tensor360, *tensor361, *tensor377, *tensor362, *tensor363, *tensor372, *tensor373;
float *tensor364, *tensor365, *tensor366, *tensor367, *tensor368, *tensor369, *tensor370, *tensor371;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][256][28][28], Gamma_beta:[512], Running_m:[256], Running_v:[256], Mu:[256], Var:[256], V1:[64][256][28][28], V2:[256], Output:[64][256][28][28]
  forwardPass_BatchNorm2d_l51<<<64, 0>>>((float (*)[256][28][28]) tensor356, (float (*)) tensor362, (float (*)) tensor372, (float (*)) tensor373, (float (*)) tensor364, (float (*)) tensor365, (float (*)[256][28][28]) tensor366, (float (*)) tensor367, (float (*)[256][28][28]) tensor360);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("052 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 308322304);
  
  // Weight addr assignment
  tensor356 = workspace_base + 0;
  tensor360 = workspace_base + 12845056;
  tensor361 = workspace_base + 25690112;
  tensor377 = workspace_base + 38535168;
  tensor362 = workspace_base + 51380224;
  tensor363 = workspace_base + 51381248;
  tensor372 = workspace_base + 51382272;
  tensor373 = workspace_base + 51383296;
  tensor364 = workspace_base + 51384320;
  tensor365 = workspace_base + 51385344;
  tensor366 = workspace_base + 51386368;
  tensor367 = workspace_base + 64231424;
  tensor368 = workspace_base + 64232448;
  tensor369 = workspace_base + 64233472;
  tensor370 = workspace_base + 64234496;
  tensor371 = workspace_base + 77079552;

  // Data & mask generation
  generate_data(tensor356, 12845056);
  generate_data(tensor360, 12845056);
  generate_data(tensor361, 12845056);
  generate_data(tensor377, 12845056);
  generate_data(tensor362, 1024);
  generate_data(tensor363, 1024);
  generate_data(tensor372, 1024);
  generate_data(tensor373, 1024);
  generate_data(tensor364, 1024);
  generate_data(tensor365, 1024);
  generate_data(tensor366, 12845056);
  generate_data(tensor367, 1024);
  generate_data(tensor368, 1024);
  generate_data(tensor369, 1024);
  generate_data(tensor370, 12845056);
  generate_data(tensor371, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
