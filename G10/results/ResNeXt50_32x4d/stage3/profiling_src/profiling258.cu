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
float *tensor616, *tensor620, *tensor621, *tensor634, *tensor622, *tensor623, *tensor632, *tensor633;
float *tensor624, *tensor625, *tensor626, *tensor627, *tensor628, *tensor629, *tensor630, *tensor631;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][1024][14][14], V1:[64][1024][14][14], V2:[1024], Gamma_beta:[2048], D_v1:[64][1024][14][14], D_v2:[1024], D_var:[1024], D_mu:[1024], D_input:[64][1024][14][14], D_gamma_beta:[2048]
  backwardPass_BatchNorm2d_l87<<<64, 0>>>((float (*)[1024][14][14]) tensor634, (float (*)[1024][14][14]) tensor626, (float (*)) tensor627, (float (*)) tensor622, (float (*)[1024][14][14]) tensor630, (float (*)) tensor631, (float (*)) tensor629, (float (*)) tensor628, (float (*)[1024][14][14]) tensor621, (float (*)) tensor623);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("258 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 308330496);
  
  // Weight addr assignment
  tensor616 = workspace_base + 0;
  tensor620 = workspace_base + 12845056;
  tensor621 = workspace_base + 25690112;
  tensor634 = workspace_base + 38535168;
  tensor622 = workspace_base + 51380224;
  tensor623 = workspace_base + 51382272;
  tensor632 = workspace_base + 51384320;
  tensor633 = workspace_base + 51385344;
  tensor624 = workspace_base + 51386368;
  tensor625 = workspace_base + 51387392;
  tensor626 = workspace_base + 51388416;
  tensor627 = workspace_base + 64233472;
  tensor628 = workspace_base + 64234496;
  tensor629 = workspace_base + 64235520;
  tensor630 = workspace_base + 64236544;
  tensor631 = workspace_base + 77081600;

  // Data & mask generation
  generate_data(tensor616, 12845056);
  generate_data(tensor620, 12845056);
  generate_data(tensor621, 12845056);
  generate_data(tensor634, 12845056);
  generate_data(tensor622, 2048);
  generate_data(tensor623, 2048);
  generate_data(tensor632, 1024);
  generate_data(tensor633, 1024);
  generate_data(tensor624, 1024);
  generate_data(tensor625, 1024);
  generate_data(tensor626, 12845056);
  generate_data(tensor627, 1024);
  generate_data(tensor628, 1024);
  generate_data(tensor629, 1024);
  generate_data(tensor630, 12845056);
  generate_data(tensor631, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
