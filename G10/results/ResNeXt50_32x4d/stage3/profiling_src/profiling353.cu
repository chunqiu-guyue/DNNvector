#include "../include/declaration5.cuh"
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
float *tensor300, *tensor304, *tensor305, *tensor321, *tensor306, *tensor307, *tensor316, *tensor317;
float *tensor308, *tensor309, *tensor310, *tensor311, *tensor312, *tensor313, *tensor314, *tensor315;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][256][28][28], V1:[64][256][28][28], V2:[256], Gamma_beta:[512], D_v1:[64][256][28][28], D_v2:[256], D_var:[256], D_mu:[256], D_input:[64][256][28][28], D_gamma_beta:[512]
  backwardPass_BatchNorm2d_l43<<<64, 0>>>((float (*)[256][28][28]) tensor321, (float (*)[256][28][28]) tensor310, (float (*)) tensor311, (float (*)) tensor306, (float (*)[256][28][28]) tensor314, (float (*)) tensor315, (float (*)) tensor313, (float (*)) tensor312, (float (*)[256][28][28]) tensor305, (float (*)) tensor307);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("353 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 308322304);
  
  // Weight addr assignment
  tensor300 = workspace_base + 0;
  tensor304 = workspace_base + 12845056;
  tensor305 = workspace_base + 25690112;
  tensor321 = workspace_base + 38535168;
  tensor306 = workspace_base + 51380224;
  tensor307 = workspace_base + 51381248;
  tensor316 = workspace_base + 51382272;
  tensor317 = workspace_base + 51383296;
  tensor308 = workspace_base + 51384320;
  tensor309 = workspace_base + 51385344;
  tensor310 = workspace_base + 51386368;
  tensor311 = workspace_base + 64231424;
  tensor312 = workspace_base + 64232448;
  tensor313 = workspace_base + 64233472;
  tensor314 = workspace_base + 64234496;
  tensor315 = workspace_base + 77079552;

  // Data & mask generation
  generate_data(tensor300, 12845056);
  generate_data(tensor304, 12845056);
  generate_data(tensor305, 12845056);
  generate_data(tensor321, 12845056);
  generate_data(tensor306, 1024);
  generate_data(tensor307, 1024);
  generate_data(tensor316, 1024);
  generate_data(tensor317, 1024);
  generate_data(tensor308, 1024);
  generate_data(tensor309, 1024);
  generate_data(tensor310, 12845056);
  generate_data(tensor311, 1024);
  generate_data(tensor312, 1024);
  generate_data(tensor313, 1024);
  generate_data(tensor314, 12845056);
  generate_data(tensor315, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
