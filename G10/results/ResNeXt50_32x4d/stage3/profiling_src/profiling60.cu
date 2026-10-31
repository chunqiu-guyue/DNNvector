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
float *tensor412, *tensor416, *tensor417, *tensor433, *tensor418, *tensor419, *tensor428, *tensor429;
float *tensor420, *tensor421, *tensor422, *tensor423, *tensor424, *tensor425, *tensor426, *tensor427;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][256][28][28], Gamma_beta:[512], Running_m:[256], Running_v:[256], Mu:[256], Var:[256], V1:[64][256][28][28], V2:[256], Output:[64][256][28][28]
  forwardPass_BatchNorm2d_l59<<<64, 0>>>((float (*)[256][28][28]) tensor412, (float (*)) tensor418, (float (*)) tensor428, (float (*)) tensor429, (float (*)) tensor420, (float (*)) tensor421, (float (*)[256][28][28]) tensor422, (float (*)) tensor423, (float (*)[256][28][28]) tensor416);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("060 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 308322304);
  
  // Weight addr assignment
  tensor412 = workspace_base + 0;
  tensor416 = workspace_base + 12845056;
  tensor417 = workspace_base + 25690112;
  tensor433 = workspace_base + 38535168;
  tensor418 = workspace_base + 51380224;
  tensor419 = workspace_base + 51381248;
  tensor428 = workspace_base + 51382272;
  tensor429 = workspace_base + 51383296;
  tensor420 = workspace_base + 51384320;
  tensor421 = workspace_base + 51385344;
  tensor422 = workspace_base + 51386368;
  tensor423 = workspace_base + 64231424;
  tensor424 = workspace_base + 64232448;
  tensor425 = workspace_base + 64233472;
  tensor426 = workspace_base + 64234496;
  tensor427 = workspace_base + 77079552;

  // Data & mask generation
  generate_data(tensor412, 12845056);
  generate_data(tensor416, 12845056);
  generate_data(tensor417, 12845056);
  generate_data(tensor433, 12845056);
  generate_data(tensor418, 1024);
  generate_data(tensor419, 1024);
  generate_data(tensor428, 1024);
  generate_data(tensor429, 1024);
  generate_data(tensor420, 1024);
  generate_data(tensor421, 1024);
  generate_data(tensor422, 12845056);
  generate_data(tensor423, 1024);
  generate_data(tensor424, 1024);
  generate_data(tensor425, 1024);
  generate_data(tensor426, 12845056);
  generate_data(tensor427, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
