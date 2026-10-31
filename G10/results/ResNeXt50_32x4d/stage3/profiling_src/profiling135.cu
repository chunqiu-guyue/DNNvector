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
float *tensor938, *tensor952, *tensor953, *tensor955, *tensor957, *tensor954;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // In:[64][1024][7][7], Out:[64][1024][7][7], Weight:[1024][1024][3][3]
  forwardPass_conv2d_l134<<<64, 0>>>((float (*)[1024][7][7]) tensor938, (float (*)[1024][7][7]) tensor952, (float (*)[1024][3][3]) tensor953);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("135 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 53739520);
  
  // Weight addr assignment
  tensor938 = workspace_base + 0;
  tensor952 = workspace_base + 3211264;
  tensor953 = workspace_base + 6422528;
  tensor955 = workspace_base + 6717440;
  tensor957 = workspace_base + 9928704;
  tensor954 = workspace_base + 13139968;

  // Data & mask generation
  generate_data(tensor938, 3211264);
  generate_data(tensor952, 3211264);
  generate_data(tensor953, 294912);
  generate_data(tensor955, 3211264);
  generate_data(tensor957, 3211264);
  generate_data(tensor954, 294912);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
