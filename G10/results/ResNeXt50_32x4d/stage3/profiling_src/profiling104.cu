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
float *tensor728, *tensor732, *tensor733, *tensor746, *tensor734, *tensor735, *tensor744, *tensor745;
float *tensor736, *tensor737, *tensor738, *tensor739, *tensor740, *tensor741, *tensor742, *tensor743;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][1024][14][14], Gamma_beta:[2048], Running_m:[1024], Running_v:[1024], Mu:[1024], Var:[1024], V1:[64][1024][14][14], V2:[1024], Output:[64][1024][14][14]
  forwardPass_BatchNorm2d_l103<<<64, 0>>>((float (*)[1024][14][14]) tensor728, (float (*)) tensor734, (float (*)) tensor744, (float (*)) tensor745, (float (*)) tensor736, (float (*)) tensor737, (float (*)[1024][14][14]) tensor738, (float (*)) tensor739, (float (*)[1024][14][14]) tensor732);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("104 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 308330496);
  
  // Weight addr assignment
  tensor728 = workspace_base + 0;
  tensor732 = workspace_base + 12845056;
  tensor733 = workspace_base + 25690112;
  tensor746 = workspace_base + 38535168;
  tensor734 = workspace_base + 51380224;
  tensor735 = workspace_base + 51382272;
  tensor744 = workspace_base + 51384320;
  tensor745 = workspace_base + 51385344;
  tensor736 = workspace_base + 51386368;
  tensor737 = workspace_base + 51387392;
  tensor738 = workspace_base + 51388416;
  tensor739 = workspace_base + 64233472;
  tensor740 = workspace_base + 64234496;
  tensor741 = workspace_base + 64235520;
  tensor742 = workspace_base + 64236544;
  tensor743 = workspace_base + 77081600;

  // Data & mask generation
  generate_data(tensor728, 12845056);
  generate_data(tensor732, 12845056);
  generate_data(tensor733, 12845056);
  generate_data(tensor746, 12845056);
  generate_data(tensor734, 2048);
  generate_data(tensor735, 2048);
  generate_data(tensor744, 1024);
  generate_data(tensor745, 1024);
  generate_data(tensor736, 1024);
  generate_data(tensor737, 1024);
  generate_data(tensor738, 12845056);
  generate_data(tensor739, 1024);
  generate_data(tensor740, 1024);
  generate_data(tensor741, 1024);
  generate_data(tensor742, 12845056);
  generate_data(tensor743, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
