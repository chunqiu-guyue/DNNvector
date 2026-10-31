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
float *tensor859, *tensor863, *tensor864, *tensor881, *tensor865, *tensor866, *tensor875, *tensor876;
float *tensor867, *tensor868, *tensor869, *tensor870, *tensor871, *tensor872, *tensor873, *tensor874;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][2048][7][7], V1:[64][2048][7][7], V2:[2048], Gamma_beta:[4096], D_v1:[64][2048][7][7], D_v2:[2048], D_var:[2048], D_mu:[2048], D_input:[64][2048][7][7], D_gamma_beta:[4096]
  backwardPass_BatchNorm2d_l122<<<64, 0>>>((float (*)[2048][7][7]) tensor881, (float (*)[2048][7][7]) tensor869, (float (*)) tensor870, (float (*)) tensor865, (float (*)[2048][7][7]) tensor873, (float (*)) tensor874, (float (*)) tensor872, (float (*)) tensor871, (float (*)[2048][7][7]) tensor864, (float (*)) tensor866);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("184 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 154238976);
  
  // Weight addr assignment
  tensor859 = workspace_base + 0;
  tensor863 = workspace_base + 6422528;
  tensor864 = workspace_base + 12845056;
  tensor881 = workspace_base + 19267584;
  tensor865 = workspace_base + 25690112;
  tensor866 = workspace_base + 25694208;
  tensor875 = workspace_base + 25698304;
  tensor876 = workspace_base + 25700352;
  tensor867 = workspace_base + 25702400;
  tensor868 = workspace_base + 25704448;
  tensor869 = workspace_base + 25706496;
  tensor870 = workspace_base + 32129024;
  tensor871 = workspace_base + 32131072;
  tensor872 = workspace_base + 32133120;
  tensor873 = workspace_base + 32135168;
  tensor874 = workspace_base + 38557696;

  // Data & mask generation
  generate_data(tensor859, 6422528);
  generate_data(tensor863, 6422528);
  generate_data(tensor864, 6422528);
  generate_data(tensor881, 6422528);
  generate_data(tensor865, 4096);
  generate_data(tensor866, 4096);
  generate_data(tensor875, 2048);
  generate_data(tensor876, 2048);
  generate_data(tensor867, 2048);
  generate_data(tensor868, 2048);
  generate_data(tensor869, 6422528);
  generate_data(tensor870, 2048);
  generate_data(tensor871, 2048);
  generate_data(tensor872, 2048);
  generate_data(tensor873, 6422528);
  generate_data(tensor874, 2048);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
