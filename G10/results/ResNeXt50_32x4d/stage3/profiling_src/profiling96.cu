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
float *tensor672, *tensor676, *tensor677, *tensor690, *tensor678, *tensor679, *tensor688, *tensor689;
float *tensor680, *tensor681, *tensor682, *tensor683, *tensor684, *tensor685, *tensor686, *tensor687;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][1024][14][14], Gamma_beta:[2048], Running_m:[1024], Running_v:[1024], Mu:[1024], Var:[1024], V1:[64][1024][14][14], V2:[1024], Output:[64][1024][14][14]
  forwardPass_BatchNorm2d_l95<<<64, 0>>>((float (*)[1024][14][14]) tensor672, (float (*)) tensor678, (float (*)) tensor688, (float (*)) tensor689, (float (*)) tensor680, (float (*)) tensor681, (float (*)[1024][14][14]) tensor682, (float (*)) tensor683, (float (*)[1024][14][14]) tensor676);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("096 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 308330496);
  
  // Weight addr assignment
  tensor672 = workspace_base + 0;
  tensor676 = workspace_base + 12845056;
  tensor677 = workspace_base + 25690112;
  tensor690 = workspace_base + 38535168;
  tensor678 = workspace_base + 51380224;
  tensor679 = workspace_base + 51382272;
  tensor688 = workspace_base + 51384320;
  tensor689 = workspace_base + 51385344;
  tensor680 = workspace_base + 51386368;
  tensor681 = workspace_base + 51387392;
  tensor682 = workspace_base + 51388416;
  tensor683 = workspace_base + 64233472;
  tensor684 = workspace_base + 64234496;
  tensor685 = workspace_base + 64235520;
  tensor686 = workspace_base + 64236544;
  tensor687 = workspace_base + 77081600;

  // Data & mask generation
  generate_data(tensor672, 12845056);
  generate_data(tensor676, 12845056);
  generate_data(tensor677, 12845056);
  generate_data(tensor690, 12845056);
  generate_data(tensor678, 2048);
  generate_data(tensor679, 2048);
  generate_data(tensor688, 1024);
  generate_data(tensor689, 1024);
  generate_data(tensor680, 1024);
  generate_data(tensor681, 1024);
  generate_data(tensor682, 12845056);
  generate_data(tensor683, 1024);
  generate_data(tensor684, 1024);
  generate_data(tensor685, 1024);
  generate_data(tensor686, 12845056);
  generate_data(tensor687, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
