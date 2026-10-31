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
float *tensor188, *tensor192, *tensor193, *tensor206, *tensor194, *tensor195, *tensor204, *tensor205;
float *tensor196, *tensor197, *tensor198, *tensor199, *tensor200, *tensor201, *tensor202, *tensor203;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][256][56][56], V1:[64][256][56][56], V2:[256], Gamma_beta:[512], D_v1:[64][256][56][56], D_v2:[256], D_var:[256], D_mu:[256], D_input:[64][256][56][56], D_gamma_beta:[512]
  backwardPass_BatchNorm2d_l27<<<64, 0>>>((float (*)[256][56][56]) tensor206, (float (*)[256][56][56]) tensor198, (float (*)) tensor199, (float (*)) tensor194, (float (*)[256][56][56]) tensor202, (float (*)) tensor203, (float (*)) tensor201, (float (*)) tensor200, (float (*)[256][56][56]) tensor193, (float (*)) tensor195);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("387 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 1233166336);
  
  // Weight addr assignment
  tensor188 = workspace_base + 0;
  tensor192 = workspace_base + 51380224;
  tensor193 = workspace_base + 102760448;
  tensor206 = workspace_base + 154140672;
  tensor194 = workspace_base + 205520896;
  tensor195 = workspace_base + 205521920;
  tensor204 = workspace_base + 205522944;
  tensor205 = workspace_base + 205523968;
  tensor196 = workspace_base + 205524992;
  tensor197 = workspace_base + 205526016;
  tensor198 = workspace_base + 205527040;
  tensor199 = workspace_base + 256907264;
  tensor200 = workspace_base + 256908288;
  tensor201 = workspace_base + 256909312;
  tensor202 = workspace_base + 256910336;
  tensor203 = workspace_base + 308290560;

  // Data & mask generation
  generate_data(tensor188, 51380224);
  generate_data(tensor192, 51380224);
  generate_data(tensor193, 51380224);
  generate_data(tensor206, 51380224);
  generate_data(tensor194, 1024);
  generate_data(tensor195, 1024);
  generate_data(tensor204, 1024);
  generate_data(tensor205, 1024);
  generate_data(tensor196, 1024);
  generate_data(tensor197, 1024);
  generate_data(tensor198, 51380224);
  generate_data(tensor199, 1024);
  generate_data(tensor200, 1024);
  generate_data(tensor201, 1024);
  generate_data(tensor202, 51380224);
  generate_data(tensor203, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
