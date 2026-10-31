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
float *tensor505, *tensor509, *tensor510, *tensor527, *tensor511, *tensor512, *tensor521, *tensor522;
float *tensor513, *tensor514, *tensor515, *tensor516, *tensor517, *tensor518, *tensor519, *tensor520;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][1024][14][14], Gamma_beta:[2048], Running_m:[1024], Running_v:[1024], Mu:[1024], Var:[1024], V1:[64][1024][14][14], V2:[1024], Output:[64][1024][14][14]
  forwardPass_BatchNorm2d_l72<<<64, 0>>>((float (*)[1024][14][14]) tensor505, (float (*)) tensor511, (float (*)) tensor521, (float (*)) tensor522, (float (*)) tensor513, (float (*)) tensor514, (float (*)[1024][14][14]) tensor515, (float (*)) tensor516, (float (*)[1024][14][14]) tensor509);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("073 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 308330496);
  
  // Weight addr assignment
  tensor505 = workspace_base + 0;
  tensor509 = workspace_base + 12845056;
  tensor510 = workspace_base + 25690112;
  tensor527 = workspace_base + 38535168;
  tensor511 = workspace_base + 51380224;
  tensor512 = workspace_base + 51382272;
  tensor521 = workspace_base + 51384320;
  tensor522 = workspace_base + 51385344;
  tensor513 = workspace_base + 51386368;
  tensor514 = workspace_base + 51387392;
  tensor515 = workspace_base + 51388416;
  tensor516 = workspace_base + 64233472;
  tensor517 = workspace_base + 64234496;
  tensor518 = workspace_base + 64235520;
  tensor519 = workspace_base + 64236544;
  tensor520 = workspace_base + 77081600;

  // Data & mask generation
  generate_data(tensor505, 12845056);
  generate_data(tensor509, 12845056);
  generate_data(tensor510, 12845056);
  generate_data(tensor527, 12845056);
  generate_data(tensor511, 2048);
  generate_data(tensor512, 2048);
  generate_data(tensor521, 1024);
  generate_data(tensor522, 1024);
  generate_data(tensor513, 1024);
  generate_data(tensor514, 1024);
  generate_data(tensor515, 12845056);
  generate_data(tensor516, 1024);
  generate_data(tensor517, 1024);
  generate_data(tensor518, 1024);
  generate_data(tensor519, 12845056);
  generate_data(tensor520, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
