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
float *tensor784, *tensor788, *tensor789, *tensor802, *tensor790, *tensor791, *tensor800, *tensor801;
float *tensor792, *tensor793, *tensor794, *tensor795, *tensor796, *tensor797, *tensor798, *tensor799;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][1024][14][14], V1:[64][1024][14][14], V2:[1024], Gamma_beta:[2048], D_v1:[64][1024][14][14], D_v2:[1024], D_var:[1024], D_mu:[1024], D_input:[64][1024][14][14], D_gamma_beta:[2048]
  backwardPass_BatchNorm2d_l111<<<64, 0>>>((float (*)[1024][14][14]) tensor802, (float (*)[1024][14][14]) tensor794, (float (*)) tensor795, (float (*)) tensor790, (float (*)[1024][14][14]) tensor798, (float (*)) tensor799, (float (*)) tensor797, (float (*)) tensor796, (float (*)[1024][14][14]) tensor789, (float (*)) tensor791);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("207 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 308330496);
  
  // Weight addr assignment
  tensor784 = workspace_base + 0;
  tensor788 = workspace_base + 12845056;
  tensor789 = workspace_base + 25690112;
  tensor802 = workspace_base + 38535168;
  tensor790 = workspace_base + 51380224;
  tensor791 = workspace_base + 51382272;
  tensor800 = workspace_base + 51384320;
  tensor801 = workspace_base + 51385344;
  tensor792 = workspace_base + 51386368;
  tensor793 = workspace_base + 51387392;
  tensor794 = workspace_base + 51388416;
  tensor795 = workspace_base + 64233472;
  tensor796 = workspace_base + 64234496;
  tensor797 = workspace_base + 64235520;
  tensor798 = workspace_base + 64236544;
  tensor799 = workspace_base + 77081600;

  // Data & mask generation
  generate_data(tensor784, 12845056);
  generate_data(tensor788, 12845056);
  generate_data(tensor789, 12845056);
  generate_data(tensor802, 12845056);
  generate_data(tensor790, 2048);
  generate_data(tensor791, 2048);
  generate_data(tensor800, 1024);
  generate_data(tensor801, 1024);
  generate_data(tensor792, 1024);
  generate_data(tensor793, 1024);
  generate_data(tensor794, 12845056);
  generate_data(tensor795, 1024);
  generate_data(tensor796, 1024);
  generate_data(tensor797, 1024);
  generate_data(tensor798, 12845056);
  generate_data(tensor799, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
