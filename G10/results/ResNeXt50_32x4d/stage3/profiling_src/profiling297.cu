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
float *tensor486, *tensor490, *tensor491, *tensor504, *tensor492, *tensor493, *tensor502, *tensor503;
float *tensor494, *tensor495, *tensor496, *tensor497, *tensor498, *tensor499, *tensor500, *tensor501;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][1024][14][14], V1:[64][1024][14][14], V2:[1024], Gamma_beta:[2048], D_v1:[64][1024][14][14], D_v2:[1024], D_var:[1024], D_mu:[1024], D_input:[64][1024][14][14], D_gamma_beta:[2048]
  backwardPass_BatchNorm2d_l69<<<64, 0>>>((float (*)[1024][14][14]) tensor504, (float (*)[1024][14][14]) tensor496, (float (*)) tensor497, (float (*)) tensor492, (float (*)[1024][14][14]) tensor500, (float (*)) tensor501, (float (*)) tensor499, (float (*)) tensor498, (float (*)[1024][14][14]) tensor491, (float (*)) tensor493);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("297 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 308330496);
  
  // Weight addr assignment
  tensor486 = workspace_base + 0;
  tensor490 = workspace_base + 12845056;
  tensor491 = workspace_base + 25690112;
  tensor504 = workspace_base + 38535168;
  tensor492 = workspace_base + 51380224;
  tensor493 = workspace_base + 51382272;
  tensor502 = workspace_base + 51384320;
  tensor503 = workspace_base + 51385344;
  tensor494 = workspace_base + 51386368;
  tensor495 = workspace_base + 51387392;
  tensor496 = workspace_base + 51388416;
  tensor497 = workspace_base + 64233472;
  tensor498 = workspace_base + 64234496;
  tensor499 = workspace_base + 64235520;
  tensor500 = workspace_base + 64236544;
  tensor501 = workspace_base + 77081600;

  // Data & mask generation
  generate_data(tensor486, 12845056);
  generate_data(tensor490, 12845056);
  generate_data(tensor491, 12845056);
  generate_data(tensor504, 12845056);
  generate_data(tensor492, 2048);
  generate_data(tensor493, 2048);
  generate_data(tensor502, 1024);
  generate_data(tensor503, 1024);
  generate_data(tensor494, 1024);
  generate_data(tensor495, 1024);
  generate_data(tensor496, 12845056);
  generate_data(tensor497, 1024);
  generate_data(tensor498, 1024);
  generate_data(tensor499, 1024);
  generate_data(tensor500, 12845056);
  generate_data(tensor501, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
