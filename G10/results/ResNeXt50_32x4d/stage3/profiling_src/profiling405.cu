#include "../include/cudadnnUtil.cuh"
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
float *tensor132, *tensor136, *tensor137, *tensor150, *tensor138, *tensor139, *tensor148, *tensor149;
float *tensor140, *tensor141, *tensor142, *tensor143, *tensor144, *tensor145, *tensor146, *tensor147;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // BatchNorm Size: 1024
  apply_grad<<<64, 0>>>(tensor138, tensor139, 1024);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("405 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 1233166336);
  
  // Weight addr assignment
  tensor132 = workspace_base + 0;
  tensor136 = workspace_base + 51380224;
  tensor137 = workspace_base + 102760448;
  tensor150 = workspace_base + 154140672;
  tensor138 = workspace_base + 205520896;
  tensor139 = workspace_base + 205521920;
  tensor148 = workspace_base + 205522944;
  tensor149 = workspace_base + 205523968;
  tensor140 = workspace_base + 205524992;
  tensor141 = workspace_base + 205526016;
  tensor142 = workspace_base + 205527040;
  tensor143 = workspace_base + 256907264;
  tensor144 = workspace_base + 256908288;
  tensor145 = workspace_base + 256909312;
  tensor146 = workspace_base + 256910336;
  tensor147 = workspace_base + 308290560;

  // Data & mask generation
  generate_data(tensor132, 51380224);
  generate_data(tensor136, 51380224);
  generate_data(tensor137, 51380224);
  generate_data(tensor150, 51380224);
  generate_data(tensor138, 1024);
  generate_data(tensor139, 1024);
  generate_data(tensor148, 1024);
  generate_data(tensor149, 1024);
  generate_data(tensor140, 1024);
  generate_data(tensor141, 1024);
  generate_data(tensor142, 51380224);
  generate_data(tensor143, 1024);
  generate_data(tensor144, 1024);
  generate_data(tensor145, 1024);
  generate_data(tensor146, 51380224);
  generate_data(tensor147, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
