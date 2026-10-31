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
float *tensor952, *tensor956, *tensor957, *tensor973, *tensor958, *tensor959, *tensor968, *tensor969;
float *tensor960, *tensor961, *tensor962, *tensor963, *tensor964, *tensor965, *tensor966, *tensor967;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][1024][7][7], V1:[64][1024][7][7], V2:[1024], Gamma_beta:[2048], D_v1:[64][1024][7][7], D_v2:[1024], D_var:[1024], D_mu:[1024], D_input:[64][1024][7][7], D_gamma_beta:[2048]
  backwardPass_BatchNorm2d_l135<<<64, 0>>>((float (*)[1024][7][7]) tensor973, (float (*)[1024][7][7]) tensor962, (float (*)) tensor963, (float (*)) tensor958, (float (*)[1024][7][7]) tensor966, (float (*)) tensor967, (float (*)) tensor965, (float (*)) tensor964, (float (*)[1024][7][7]) tensor957, (float (*)) tensor959);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("156 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 77119488);
  
  // Weight addr assignment
  tensor952 = workspace_base + 0;
  tensor956 = workspace_base + 3211264;
  tensor957 = workspace_base + 6422528;
  tensor973 = workspace_base + 9633792;
  tensor958 = workspace_base + 12845056;
  tensor959 = workspace_base + 12847104;
  tensor968 = workspace_base + 12849152;
  tensor969 = workspace_base + 12850176;
  tensor960 = workspace_base + 12851200;
  tensor961 = workspace_base + 12852224;
  tensor962 = workspace_base + 12853248;
  tensor963 = workspace_base + 16064512;
  tensor964 = workspace_base + 16065536;
  tensor965 = workspace_base + 16066560;
  tensor966 = workspace_base + 16067584;
  tensor967 = workspace_base + 19278848;

  // Data & mask generation
  generate_data(tensor952, 3211264);
  generate_data(tensor956, 3211264);
  generate_data(tensor957, 3211264);
  generate_data(tensor973, 3211264);
  generate_data(tensor958, 2048);
  generate_data(tensor959, 2048);
  generate_data(tensor968, 1024);
  generate_data(tensor969, 1024);
  generate_data(tensor960, 1024);
  generate_data(tensor961, 1024);
  generate_data(tensor962, 3211264);
  generate_data(tensor963, 1024);
  generate_data(tensor964, 1024);
  generate_data(tensor965, 1024);
  generate_data(tensor966, 3211264);
  generate_data(tensor967, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
