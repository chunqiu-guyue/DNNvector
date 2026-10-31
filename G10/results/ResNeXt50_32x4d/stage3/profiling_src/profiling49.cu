#include "../include/declaration0.cuh"
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
float *tensor337, *tensor338, *tensor339, *tensor341, *tensor343, *tensor340;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][512][28][28], Out:[64][256][28][28], Weight:[256][512][1][1]
  forwardPass_conv2d_l48<<<64, 0>>>((float (*)[512][28][28]) tensor337, (float (*)[256][28][28]) tensor338, (float (*)[512][1][1]) tensor339);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("049 %20.10f ms\n", elapsedTime);
}

0// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 309329920);
  
  // Weight addr assignment
  tensor337 = workspace_base + 0;
  tensor338 = workspace_base + 25690112;
  tensor339 = workspace_base + 38535168;
  tensor341 = workspace_base + 38666240;
  tensor343 = workspace_base + 64356352;
  tensor340 = workspace_base + 77201408;

  // Data & mask generation
  generate_data(tensor337, 25690112);
  generate_data(tensor338, 12845056);
  generate_data(tensor339, 131072);
  generate_data(tensor341, 25690112);
  generate_data(tensor343, 12845056);
  generate_data(tensor340, 131072);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
